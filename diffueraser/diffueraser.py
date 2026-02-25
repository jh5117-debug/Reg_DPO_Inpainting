# -*- coding: utf-8 -*-
"""
DiffuEraser module - Modified to support returning frames directly.
This avoids video compression loss when computing metrics.

Add return_frames=True to forward() to get list of RGB numpy arrays.
"""
import gc
import copy
import cv2
import os
import numpy as np
import scipy.ndimage
import torch
import torchvision
from einops import repeat
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig

from libs.unet_motion_model import MotionAdapter, UNetMotionModel
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        4,
        0.0,
    ],
}

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames

def read_mask(validation_mask, fps, n_total_frames, img_size, mask_dilation_iter, frames):
    """
    Read masks losslessly from:
      - mask video file (legacy)
      - directory of mask images (recommended; lossless)

    IMPORTANT:
      - For alignment, we do NOT enforce FPS when using directories.
      - Mask resize, if needed, uses NEAREST.
    """
    masks = []
    if os.path.isdir(validation_mask):
        mnames = sorted([f for f in os.listdir(validation_mask) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for mn in mnames:
            m = cv2.imread(os.path.join(validation_mask, mn), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            masks.append(m)
        if len(masks) == 0:
            raise ValueError(f"No masks found in directory: {validation_mask}")
        # length normalize
        if len(masks) < n_total_frames:
            masks = masks + [masks[-1]] * (n_total_frames - len(masks))
        if len(masks) > n_total_frames:
            masks = masks[:n_total_frames]
    else:
        cap = cv2.VideoCapture(validation_mask)
        if not cap.isOpened():
            raise ValueError(f"Could not open mask video: {validation_mask}")
        mask_fps = cap.get(cv2.CAP_PROP_FPS)
        if abs(mask_fps - fps) > 1e-3:
            cap.release()
            raise ValueError("The frame rate of all input videos needs to be consistent.")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or idx >= n_total_frames:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append(frame)
            idx += 1
        cap.release()
        if len(masks) < n_total_frames:
            masks = masks + [masks[-1]] * (n_total_frames - len(masks))

    masks_pil = []
    masked_images = []
    for i in range(n_total_frames):
        m = masks[i]
        if m.shape[1] != img_size[0] or m.shape[0] != img_size[1]:
            m = cv2.resize(m, img_size, interpolation=cv2.INTER_NEAREST)
        m01 = (m > 0).astype(np.uint8)
        if mask_dilation_iter and mask_dilation_iter > 0:
            m01 = scipy.ndimage.binary_dilation(m01, iterations=int(mask_dilation_iter)).astype(np.uint8)
        masks_pil.append(Image.fromarray(m01 * 255))
        # masked image for display / debug
        fr = np.array(frames[i]).astype(np.uint8)
        fr_masked = fr.copy()
        fr_masked[m01 > 0] = 0
        masked_images.append(Image.fromarray(fr_masked))
    return masks_pil, masked_images

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
    if priori_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    os.remove(priori) # remove priori 

    return prioris


def read_priori_from_frames(priori_frames, img_size):
    """Read priori from list of RGB numpy arrays (lossless, strict size)."""
    prioris = []
    for frame in priori_frames:
        if not isinstance(frame, np.ndarray):
            raise TypeError("priori_frames must be a list of numpy arrays")
        img = Image.fromarray(frame.astype(np.uint8))
        if img.size != img_size:
            raise ValueError(f"Prior frame size {img.size} != expected {img_size} (strict no-resize).")
        prioris.append(img)
    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    """
    Read video frames from:
      - an MP4/AVI file (legacy)
      - a directory containing frames (recommended; lossless)

    IMPORTANT ALIGNMENT:
      - video_length is interpreted as MAX FRAMES (<=0 => full length), NOT seconds.
    """
    frames = []
    fps = 24.0

    if os.path.isdir(validation_image):
        fr_lst = sorted([f for f in os.listdir(validation_image) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fr in fr_lst:
            img = cv2.imread(os.path.join(validation_image, fr))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        if len(frames) == 0:
            raise ValueError(f"No frames found in directory: {validation_image}")
        if isinstance(video_length, (int, float)) and int(video_length) > 0:
            frames = frames[:int(video_length)]
    else:
        # fallback: file input. We read full video then truncate by frames.
        vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec')
        fps = float(info.get('video_fps', 24.0))
        frames = [Image.fromarray(f.numpy() if hasattr(f, 'numpy') else f) for f in vframes]
        if isinstance(video_length, (int, float)) and int(video_length) > 0:
            frames = frames[:int(video_length)]

    max_size = max(frames[0].size)
    if max_size < 256:
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if max_size > 4096:
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")

    # size policy: only resize if exceeding max_img_size or not divisible by 8
    resize_flag = False
    if max_size > max_img_size:
        ratio = max_size / max_img_size
        ratio_size = (int(frames[0].size[0] / ratio), int(frames[0].size[1] / ratio))
        img_size = (ratio_size[0] - ratio_size[0] % 8, ratio_size[1] - ratio_size[1] % 8)
        resize_flag = True
    elif (frames[0].size[0] % 8 == 0) and (frames[0].size[1] % 8 == 0):
        img_size = frames[0].size
        resize_flag = False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0] - ratio_size[0] % 8, ratio_size[1] - ratio_size[1] % 8)
        resize_flag = True

    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    n_total_frames = len(frames)
    n_clip = int(np.ceil(n_total_frames / nframes)) if nframes > 0 else 1

    return frames, fps, img_size, n_clip, n_total_frames

class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="2-Step", mode="sd15", loaded=None, pcm_weights_path="weights/PCM_Weights"):
        self.device = device
        self.pcm_weights_path = pcm_weights_path

        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, 
                subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    subfolder="tokenizer",
                    use_fast=False,
                )
        text_encoder_cls = import_model_class_from_model_name_or_path(base_model_path,revision)
        self.text_encoder = text_encoder_cls.from_pretrained(
                base_model_path, subfolder="text_encoder"
            )
        self.brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet")
        self.unet_main = UNetMotionModel.from_pretrained(
            diffueraser_path, subfolder="unet_main",
        )

        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            base_model_path,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet
        ).to(self.device, torch.float16)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        self.noise_scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            self.pipeline.load_lora_weights(
                pcm_weights_path, weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]

    def forward(self, validation_image, validation_mask, priori, output_path,
                max_img_size=1280, video_length=2, mask_dilation_iter=0,
                nframes=22, seed=None, revision=None, guidance_scale=None, blended=True,
                priori_frames=None, return_frames=False, anchor_frame=None,
                keep_anchor=False, prompt="", n_prompt=""):
        """
        Args:
            priori: Path to priori video (will be deleted after reading)
            priori_frames: Optional list of RGB numpy arrays (use this to avoid video compression loss)
            return_frames: If True, return list of RGB numpy arrays instead of just saving video
            anchor_frame: Optional PIL Image (RGB) — a pre-inpainted anchor frame.
                          If provided, it is prepended to the video sequence as additional
                          texture guidance (via temporal attention).
            keep_anchor: If True, the anchor frame's diffusion output REPLACES the
                         corresponding frame (index 0) in the final output instead of
                         being discarded. This preserves the anchor's high-quality
                         reconstruction in the output video.
        """
        validation_prompt = prompt if prompt else ""
        validation_n_prompt = n_prompt if n_prompt else ""
        guidance_scale_final = self.guidance_scale if guidance_scale==None else guidance_scale

        if (max_img_size<256 or max_img_size>1920):
            raise ValueError("The max_img_size must be larger than 256, smaller than 1920.")

        frames, fps, img_size, n_clip, n_total_frames = read_video(validation_image, video_length, nframes, max_img_size)
        video_len = len(frames)

        validation_masks_input, validation_images_input = read_mask(validation_mask, fps, video_len, img_size, mask_dilation_iter, frames)
  
        # Read priori from frames if provided, otherwise from video file
        if priori_frames is not None:
            prioris = read_priori_from_frames(priori_frames, img_size)
        else:
            prioris = read_priori(priori, fps, n_total_frames, img_size)

        n_total_frames = min(min(len(frames), len(validation_masks_input)), len(prioris))
        if(n_total_frames<22):
            raise ValueError("The effective video duration is too short. Please make sure that the number of frames of video, mask, and priori is at least greater than 22 frames.")
        validation_masks_input = validation_masks_input[:n_total_frames]
        validation_images_input = validation_images_input[:n_total_frames]
        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]

        prioris = resize_frames(prioris)
        validation_masks_input = resize_frames(validation_masks_input)
        validation_images_input = resize_frames(validation_images_input)
        resized_frames = resize_frames(frames)

        print("DiffuEraser inference...")
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        real_video_length = len(validation_images_input)
        tar_width, tar_height = validation_images_input[0].size 
        shape = (
            nframes,
            4,
            tar_height//8,
            tar_width//8
        )
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet_main is not None:
            prompt_embeds_dtype = self.unet_main.dtype
        else:
            prompt_embeds_dtype = torch.float16
        noise_pre = randn_tensor(shape, device=torch.device(self.device), dtype=prompt_embeds_dtype, generator=generator) 
        noise = repeat(noise_pre, "t c h w->(repeat t) c h w", repeat=n_clip)[:real_video_length,...]
        
        images_preprocessed = []
        for image in prioris:
            image = self.image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
            image = image.to(device=torch.device(self.device), dtype=torch.float16)
            images_preprocessed.append(image)
        pixel_values = torch.cat(images_preprocessed)

        with torch.no_grad():
            pixel_values = pixel_values.to(dtype=torch.float16)
            latents = []
            num=4
            for i in range(0, pixel_values.shape[0], num):
                latents.append(self.vae.encode(pixel_values[i : i + num]).latent_dist.sample())
            latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        torch.cuda.empty_cache()  
        timesteps = torch.tensor([0], device=self.device)
        timesteps = timesteps.long()

        validation_masks_input_ori = copy.deepcopy(validation_masks_input)
        resized_frames_ori = copy.deepcopy(resized_frames)
        
        if n_total_frames > nframes*2:
            step = n_total_frames / nframes
            sample_index = [int(i * step) for i in range(nframes)]
            sample_index = sample_index[:22]
            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre, timesteps) 
            latents_pre = noisy_latents_pre

            with torch.no_grad():
                latents_pre_out = self.pipeline(
                    num_frames=nframes, 
                    prompt=validation_prompt, 
                    negative_prompt=validation_n_prompt if validation_n_prompt else None,
                    images=validation_images_input_pre, 
                    masks=validation_masks_input_pre, 
                    num_inference_steps=self.num_inference_steps, 
                    generator=generator,
                    guidance_scale=guidance_scale_final,
                    latents=latents_pre,
                ).latents
            torch.cuda.empty_cache()  

            def decode_latents(latents, weight_dtype):
                latents = 1 / self.vae.config.scaling_factor * latents
                video = []
                for t in range(latents.shape[0]):
                    video.append(self.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
                video = torch.concat(video, dim=0)
                video = video.float()
                return video
            with torch.no_grad():
                video_tensor_temp = decode_latents(latents_pre_out, weight_dtype=torch.float16)
                images_pre_out  = self.image_processor.postprocess(video_tensor_temp, output_type="pil")
            torch.cuda.empty_cache()  

            black_image = Image.new('L', validation_masks_input[0].size, color=0)
            for i,index in enumerate(sample_index):
                latents[index] = latents_pre_out[i]
                validation_masks_input[index] = black_image
                validation_images_input[index] = images_pre_out[i]
                resized_frames[index] = images_pre_out[i]
        else:
            latents_pre_out=None
            sample_index=None
        gc.collect()
        torch.cuda.empty_cache()

        # ============== Anchor Frame Injection ==============
        # Prepend a pre-inpainted anchor frame to the sequence.
        # The anchor provides clean texture guidance via temporal attention
        # during denoising, then is discarded from the final output.
        _anchor_offset = 0
        if anchor_frame is not None:
            _anchor_offset = 1
            print("  [Anchor] Injecting pre-inpainted anchor frame as texture guidance...")

            # Ensure anchor_frame is a PIL Image at the same size as video frames
            if isinstance(anchor_frame, np.ndarray):
                anchor_pil = Image.fromarray(anchor_frame.astype(np.uint8))
            else:
                anchor_pil = anchor_frame
            anchor_pil = resize_frames([anchor_pil], (tar_width, tar_height))[0]

            # Anchor mask: all black = no hole (frame is already clean)
            anchor_mask_pil = Image.new('L', anchor_pil.size, color=0)

            # Encode anchor frame to latent space
            anchor_preprocessed = self.image_processor.preprocess(
                anchor_pil, height=tar_height, width=tar_width
            ).to(dtype=torch.float32)
            anchor_preprocessed = anchor_preprocessed.to(
                device=torch.device(self.device), dtype=torch.float16
            )
            with torch.no_grad():
                anchor_latent = self.vae.encode(anchor_preprocessed).latent_dist.sample()
            anchor_latent = anchor_latent * self.vae.config.scaling_factor  # (1,4,H/8,W/8)

            # Generate noise for the anchor (same minimal timestep as other frames)
            anchor_noise = randn_tensor(
                (1, 4, tar_height // 8, tar_width // 8),
                device=torch.device(self.device),
                dtype=prompt_embeds_dtype,
                generator=generator,
            )

            # Prepend anchor to all sequences
            latents = torch.cat([anchor_latent, latents], dim=0)
            noise = torch.cat([anchor_noise, noise], dim=0)
            validation_masks_input.insert(0, anchor_mask_pil)
            validation_images_input.insert(0, anchor_pil)
            resized_frames.insert(0, anchor_pil)
            real_video_length += 1

            torch.cuda.empty_cache()
        # ============== End Anchor Injection ==============

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) 
        latents = noisy_latents
        with torch.no_grad():
            images = self.pipeline(
                num_frames=nframes, 
                prompt=validation_prompt, 
                negative_prompt=validation_n_prompt if validation_n_prompt else None,
                images=validation_images_input, 
                masks=validation_masks_input, 
                num_inference_steps=self.num_inference_steps, 
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents,
            ).frames
        images = images[:real_video_length]

        # Handle anchor frame in the output
        if _anchor_offset > 0:
            if keep_anchor:
                # Keep anchor output as frame 0, drop the duplicate original frame 0's output
                # images[0] = anchor output, images[1] = original frame 0 output, images[2] = frame 1, ...
                # Result: [anchor_output, frame1, frame2, ...] — same length as original video
                anchor_output = images[0]
                images = [anchor_output] + list(images[_anchor_offset + 1:])
                real_video_length -= _anchor_offset  # compensate for the extra frame
                print(f"  [Anchor] Keeping anchor output as frame 0 (replacing duplicate)")
            else:
                # Original behavior: strip the anchor frame entirely
                images = images[_anchor_offset:]
                real_video_length -= _anchor_offset

        gc.collect()
        torch.cuda.empty_cache()

        binary_masks = validation_masks_input_ori
        # NOTE: blended GaussianBlur removed — hard binary mask gives best PSNR/SSIM
        
        comp_frames = []
        for i in range(len(images)):
            mask = np.expand_dims(np.array(binary_masks[i]),2).repeat(3, axis=2).astype(np.float32)/255.
            img = (np.array(images[i]).astype(np.uint8) * mask \
                + np.array(resized_frames_ori[i]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
            comp_frames.append(Image.fromarray(img))

        # Save video if output_path is provided
        if output_path is not None:
            default_fps = fps
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                default_fps, comp_frames[0].size)
            for f in range(real_video_length):
                img = np.array(comp_frames[f]).astype(np.uint8)
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()

        # Return frames if requested (for metrics computation without video compression loss)
        if return_frames:
            return [np.array(f).astype(np.uint8) for f in comp_frames]  # List of RGB numpy arrays
        else:
            return output_path
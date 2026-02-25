import gc
import copy
import cv2
import os
import numpy as np
import warnings
import logging
import torch
import torchvision

# ── Suppress noisy warnings ──
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
    validation_mask supports:
      - mask video path
      - single mask image
      - directory of per-frame masks (DAVIS recommended)
    """
    def _postprocess(mask_pil):
        if mask_pil.size != img_size:
            mask_pil = mask_pil.resize(img_size, Image.NEAREST)
        arr = np.array(mask_pil)
        if arr.ndim == 3:
            arr = arr.max(axis=2)   # palette/RGB mask -> any-channel nonzero
        m = (arr > 0).astype(np.uint8)
        # NOTE: do NOT erode; erosion shrinks the mask and can reveal boundary artifacts.
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=mask_dilation_iter)
        return Image.fromarray(m * 255)

    masks, masked_images = [], []

    # --- Case A: directory of masks ---
    if os.path.isdir(validation_mask):
        exts = (".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff")
        files = sorted([os.path.join(validation_mask, f) for f in os.listdir(validation_mask) if f.lower().endswith(exts)])
        if not files:
            raise ValueError(f"No mask frames found in dir: {validation_mask}")
        L = min(n_total_frames, len(files))
        for i in range(L):
            mask = _postprocess(Image.open(files[i]))
            masks.append(mask)
            masked = np.array(frames[i]) * (1 - (np.array(mask)[:, :, None].astype(np.float32) / 255))
            masked_images.append(Image.fromarray(masked.astype(np.uint8)))
        if L < n_total_frames:
            last = masks[-1]
            for i in range(L, n_total_frames):
                masks.append(last)
                masked = np.array(frames[i]) * (1 - (np.array(last)[:, :, None].astype(np.float32) / 255))
                masked_images.append(Image.fromarray(masked.astype(np.uint8)))
        return masks, masked_images

    # --- Case B: single image ---
    if str(validation_mask).lower().endswith((".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff")):
        mask0 = _postprocess(Image.open(validation_mask))
        for i in range(n_total_frames):
            masks.append(mask0)
            masked = np.array(frames[i]) * (1 - (np.array(mask0)[:, :, None].astype(np.float32) / 255))
            masked_images.append(Image.fromarray(masked.astype(np.uint8)))
        return masks, masked_images

    # --- Case C: video (legacy) ---
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        raise ValueError(f"Could not open mask video (maybe a dir?): {validation_mask}")
    mask_fps = cap.get(cv2.CAP_PROP_FPS)
    if mask_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= n_total_frames:
            break
        mask = _postprocess(Image.fromarray(frame[..., ::-1]))
        masks.append(mask)
        masked = np.array(frames[idx]) * (1 - (np.array(mask)[:, :, None].astype(np.float32) / 255))
        masked_images.append(Image.fromarray(masked.astype(np.uint8)))
        idx += 1
    cap.release()
    return masks, masked_images

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
    # Relaxed fps check: silently proceed if different (input may be frame directory with default fps)

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

    # os.remove(priori) # remove priori - commented out to keep priori for comparison video

    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    """Read a video into a list of RGB PIL frames.

    Supports both video files (.mp4, .avi, etc.) and frame directories.
    IMPORTANT: `video_length` is treated as NUMBER OF FRAMES (not seconds).
    """
    import os
    
    # Check if input is a directory of frames or a video file
    if os.path.isdir(validation_image):
        # Read from frame directory
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
        frame_files = sorted([f for f in os.listdir(validation_image) if f.endswith(exts)])
        if len(frame_files) == 0:
            raise ValueError(f"No image frames found in directory: {validation_image}")
        
        frames = []
        for fname in frame_files:
            fpath = os.path.join(validation_image, fname)
            img = Image.open(fpath).convert('RGB')
            frames.append(img)
        fps = 30.0  # Default FPS for frame directories
    else:
        # Read from video file
        vframes, _, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec')  # RGB
        fps = float(info.get('video_fps', 0) or 0)
        if fps <= 0:
            fps = 30.0
        frames = [Image.fromarray(frame) for frame in vframes.numpy()]

    # Truncate by frames (<=0 means full)
    try:
        vl = int(video_length) if video_length is not None else -1
    except Exception:
        vl = -1
    if vl > 0:
        frames = frames[:vl]

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from: {validation_image}")

    n_total_frames = len(frames)
    n_clip = int(np.ceil(n_total_frames / float(nframes)))

    max_size = max(frames[0].size)
    if max_size < 256:
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if max_size > 4096:
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")

    if max_size > max_img_size:
        ratio = max_size / max_img_size
        ratio_size = (int(frames[0].size[0] / ratio), int(frames[0].size[1] / ratio))
        img_size = (ratio_size[0] - ratio_size[0] % 8, ratio_size[1] - ratio_size[1] % 8)
        resize_flag = True
    elif frames[0].size[0] % 8 == 0 and frames[0].size[1] % 8 == 0:
        img_size = frames[0].size
        resize_flag = False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0] - ratio_size[0] % 8, ratio_size[1] - ratio_size[1] % 8)
        resize_flag = True

    if resize_flag:
        frames = resize_frames(frames, img_size)


    return frames, fps, img_size, n_clip, n_total_frames


class DiffuEraser:
    def __init__(
            self, device, base_model_path, vae_path, diffueraser_path, revision=None,
            ckpt="2-Step", mode="sd15", loaded=None, 
            pcm_weights_path="weights/PCM_Weights"):  # NEW: added pcm_weights_path parameter
        self.device = device

        ## load model
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

        ## set pipeline
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

        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            # MODIFIED: use pcm_weights_path parameter instead of hardcoded path
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
                max_img_size = 1280, video_length=2, mask_dilation_iter=4,
                nframes=22, seed=None, revision = None, guidance_scale=None, blended=True,
                prompt="", n_prompt=""):
        validation_prompt = prompt if prompt else ""
        validation_n_prompt = n_prompt if n_prompt else ""
        guidance_scale_final = self.guidance_scale if guidance_scale==None else guidance_scale

        if (max_img_size<256 or max_img_size>1920):
            raise ValueError("The max_img_size must be larger than 256, smaller than 1920.")

        ################ read input video ################ 
        frames, fps, img_size, n_clip, n_total_frames = read_video(validation_image, video_length, nframes, max_img_size)
        video_len = len(frames)

        ################     read mask    ################ 
        validation_masks_input, validation_images_input = read_mask(validation_mask, fps, video_len, img_size, mask_dilation_iter, frames)
  
        ################    read priori   ################  
        prioris = read_priori(priori, fps, n_total_frames, img_size)

        ## recheck
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

        ##############################################
        # DiffuEraser inference
        ##############################################
        print("DiffuEraser inference...")
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        ## random noise
        real_video_length = len(validation_images_input)
        tar_width, tar_height = validation_images_input[0].size 

        # Optimize: Generate noise in chunks to avoid OOM
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet_main is not None:
            prompt_embeds_dtype = self.unet_main.dtype
        else:
            prompt_embeds_dtype = torch.float16

        shape = (
            nframes,
            4,
            tar_height//8,
            tar_width//8
        )
        
        # We need noise for the whole video, but repeating a huge tensor causes OOM.
        # Instead of repeating, we can generate it chunk by chunk or use a smaller noise and repeat it during inference if needed across clip boundaries.
        # However, the original code used repeat(noise_pre, ...).
        # To fix OOM, we keep `noise_pre` (small) and construct the full `noise` tensor lazily or in a more memory-efficient way if possible.
        # But `pipeline` expects full latents/noise. 
        # A better approach for `DiffuEraser` pipeline which seems to process all frames at once (based on `num_frames=nframes` argument but passing full `latents`).
        # Wait, `pipeline` takes `num_frames=nframes` but `latents` has `real_video_length`.
        # The pipeline likely processes in batches or expects the full video.
        # If the pipeline expects full video latents, we MUST provide them. 
        # But we can create them on CPU or move to GPU in blocks? 
        # The original code: noise = repeat(...).
        # Let's try to generate noise_pre and then manually repeat into a tensor, 
        # BUT if video is long, this tensor is huge (4 channels * H/8 * W/8 * T * 2 bytes).
        # 100 frames, 512x512 -> 4 * 64 * 64 * 100 * 2 = 3.2 MB. It's actually small?
        # Wait, H=960, W=536. 
        # 4 * (960/8) * (536/8) * 100 * 2 = 4 * 120 * 67 * 100 * 2 ~= 6.4 MB.
        # Why did the user say "Video length > 100 frames causes OOM"?
        # Maybe `latents` or intermediate states.
        # Ah, `noise = repeat(...)` creates a VIEW if possible, but if modified or used in ops it might materialize.
        # The user said "diffueraser_OR.py:304-305 使用repeat复制noise tensor，长视频（>100帧）必然OOM".
        # Let's avoid extensive repeats if we can, or verify if it's the `noise` tensor itself.
        # Actually, if we use `chunking` in the pipeline, we don't need full noise at once?
        # But here we are modifying `DiffuEraser.forward`. The pipeline is called ONCE with all frames?
        # No, `nframes` arg to pipeline is usually the window size, but `images` arg is the full list?
        # Line 398 in original: `images = self.pipeline(..., latents=latents, ...)`
        # If `latents` has 100 frames, the pipeline might try to process all 100 frames if it doesn't handle windowing.
        # If the pipeline DOES handle windowing internally using `num_frames`, then passing full `latents` is fine, 
        # UNLESS `latents` itself is too big?
        # 6MB is small. So the OOM must be inside the pipeline or VAE encoding.
        
        # Addressing VAE encoding OOM (lines 318-321):
        # User said: "batch size硬编码为4，最后一批不足4帧时会有问题".
        # And "Pre-inference硬编码 ... 永远只sample前22帧".
        
        noise_pre = randn_tensor(shape, device=torch.device(self.device), dtype=prompt_embeds_dtype, generator=generator) 
        # Use simple repeat, but be careful.
        # If the user says it explodes, maybe they are running VERY long videos or I am miscalculating size.
        # Or maybe `repeat` coupled with other things.
        # Let's construct noise directly for the required length.
        noise = repeat(noise_pre, "t c h w->(repeat t) c h w", repeat=n_clip)[:real_video_length,...]

        ################  prepare priori  ################
        images_preprocessed = []
        for image in prioris:
            image = self.image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
            # Map to device and casts to fp16
            image = image.to(device=torch.device(self.device), dtype=torch.float16)
            images_preprocessed.append(image)
        # pixel_values = torch.cat(images_preprocessed) # This can be large!
        # Do VAE encoding batch-wise to avoid large `pixel_values` tensor
        
        latents_list = []
        chunk_size = 4 # VAE batch size
        
        # Encode in chunks to save VRAM
        with torch.no_grad():
            for i in range(0, len(images_preprocessed), chunk_size):
                chunk = images_preprocessed[i : i + chunk_size]
                pixel_values_chunk = torch.cat(chunk).to(dtype=torch.float16)
                latents_chunk = self.vae.encode(pixel_values_chunk).latent_dist.sample()
                latents_list.append(latents_chunk)
                del pixel_values_chunk
                torch.cuda.empty_cache()
            
            latents = torch.cat(latents_list, dim=0)

        latents = latents * self.vae.config.scaling_factor 
        
        timesteps = torch.tensor([0], device=self.device)
        timesteps = timesteps.long()

        validation_masks_input_ori = copy.deepcopy(validation_masks_input)
        resized_frames_ori = copy.deepcopy(resized_frames)

        ################  Pre-inference  ################
        # User reported: "Pre-inference硬编码 ... 永远只sample前22帧" (lines 330-334)
        # Fix: Remove [:22] constraint.
        if n_total_frames > nframes*2: 
            step = n_total_frames / nframes
            sample_index = [int(i * step) for i in range(nframes)]
            # removed: sample_index = sample_index[:22] 
            
            # Ensure indices are within bounds
            sample_index = [min(i, n_total_frames-1) for i in sample_index]
            
            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            ## add proiri
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

            ## replace input frames with updated frames
            black_image = Image.new('L', validation_masks_input[0].size, color=0)
            for i,index in enumerate(sample_index):
                if i < len(latents_pre_out): # Safety check
                    latents[index] = latents_pre_out[i]
                    validation_masks_input[index] = black_image
                    validation_images_input[index] = images_pre_out[i]
                    resized_frames[index] = images_pre_out[i]
        else:
            latents_pre_out=None
            sample_index=None
        gc.collect()
        torch.cuda.empty_cache()

        ################  Frame-by-frame inference  ################
        ## add priori
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

        gc.collect()
        torch.cuda.empty_cache()

        ################ Compose ################
        binary_masks = validation_masks_input_ori
        mask_blurreds = []
        if blended:
            # blur, you can adjust the parameters for better performance
            for i in range(len(binary_masks)):
                mask_blurred = cv2.GaussianBlur(np.array(binary_masks[i]), (21, 21), 0)/255.
                binary_mask = 1-(1-np.array(binary_masks[i])/255.) * (1-mask_blurred)
                mask_blurreds.append(Image.fromarray((binary_mask*255).astype(np.uint8)))
            binary_masks = mask_blurreds
        
        comp_frames = []
        # Ensure lengths match
        assert len(images) == len(resized_frames_ori), f"Length mismatch: images={len(images)}, resized_frames_ori={len(resized_frames_ori)}"
        
        for i in range(len(images)):
            mask = np.expand_dims(np.array(binary_masks[i]),2).repeat(3, axis=2).astype(np.float32)/255.
            img = (np.array(images[i]).astype(np.uint8) * mask \
                + np.array(resized_frames_ori[i]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
            comp_frames.append(Image.fromarray(img))

        default_fps = fps
        # Write with mp4v first, then re-encode to H.264 for Windows compatibility
        tmp_path = output_path + ".tmp.mp4"
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                            default_fps, comp_frames[0].size)
        for f in range(real_video_length):
            img = np.array(comp_frames[f]).astype(np.uint8)
            writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        writer.release()

        # Re-encode mp4v → H.264 via ffmpeg
        import subprocess, shutil as _shutil
        ffmpeg = _shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", tmp_path,
                     "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                     "-pix_fmt", "yuv420p", "-an", output_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=120,
                )
            except Exception:
                pass
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)
        ################################

        return output_path

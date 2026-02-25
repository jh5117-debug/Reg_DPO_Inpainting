# -*- coding: utf-8 -*-
"""
Metrics calculation module for video inpainting evaluation.
Self-contained version with built-in I3D model (no external dependencies on ProPainter).
Supports: PSNR, SSIM, LPIPS, AS, IS, VFID
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
import cv2

# =========================
# PSNR
# =========================
def compute_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio). Input: ndarray [0,255]"""
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))

# =========================
# SSIM
# =========================
def compute_ssim(img1, img2):
    """Calculate SSIM. Input: ndarray [0,255], shape (H, W, 3)"""
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    try:
        from skimage import measure
        return float(measure.compare_ssim(img1, img2, data_range=255, multichannel=True, win_size=65))
    except (AttributeError, ImportError):
        from skimage.metrics import structural_similarity
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(65, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(3, win_size)
        return float(structural_similarity(img1, img2, data_range=255, channel_axis=-1, win_size=win_size))

def calc_psnr_and_ssim(img1, img2):
    """Calculate PSNR and SSIM for images."""
    return compute_psnr(img1, img2), compute_ssim(img1, img2)

# =========================
# LPIPS
# =========================
class LPIPSMetric:
    _instance, _device = None, None
    
    @classmethod
    def get_instance(cls, device="cuda"):
        if cls._instance is None or cls._device != device:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            cls._instance = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device).eval()
            cls._device = device
        return cls._instance
    
    @staticmethod
    @torch.no_grad()
    def compute(img_gt, img_pred, device="cuda"):
        model = LPIPSMetric.get_instance(device)
        if isinstance(img_gt, Image.Image): img_gt = np.array(img_gt)
        if isinstance(img_pred, Image.Image): img_pred = np.array(img_pred)
        
        gt = torch.from_numpy(img_gt.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        pr = torch.from_numpy(img_pred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        return float(model(pr * 2.0 - 1.0, gt * 2.0 - 1.0).detach().cpu().item())

# =========================
# AS (Aesthetic Score)
# =========================
class AestheticScoreMetric:
    _instance, _device = None, None
    
    @classmethod
    def get_instance(cls, device="cuda", ckpt_path="/home/hj/DiffuEraser1/weights/metrics/sa_0_4_vit_l_14_linear.pth"):
        if cls._instance is None or cls._device != device:
            import open_clip
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            clip_model = clip_model.to(device).eval()
            head = torch.nn.Linear(768, 1).to(device)
            
            if os.path.exists(ckpt_path):
                head.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f"[AS] Checkpoint not found, downloading...")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                from urllib.request import urlretrieve
                urlretrieve("https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true", ckpt_path)
                head.load_state_dict(torch.load(ckpt_path, map_location=device))
            
            head.eval()
            cls._instance = (clip_model, clip_preprocess, head)
            cls._device = device
        return cls._instance
    
    @staticmethod
    @torch.no_grad()
    def compute(pil_img, device="cuda", ckpt_path="/home/hj/DiffuEraser1/weights/metrics/sa_0_4_vit_l_14_linear.pth"):
        clip_model, clip_preprocess, head = AestheticScoreMetric.get_instance(device, ckpt_path)
        if isinstance(pil_img, np.ndarray):
            pil_img = Image.fromarray(pil_img.astype(np.uint8))
        feat = clip_model.encode_image(clip_preprocess(pil_img).unsqueeze(0).to(device))
        return float(head(feat / feat.norm(dim=-1, keepdim=True).float()).detach().cpu().item())

# =========================
# IS (Inception Score)
# =========================
class InceptionScoreMetric:
    _instance, _device = None, None
    
    @classmethod
    def get_instance(cls, device="cuda"):
        if cls._instance is None or cls._device != device:
            from torchvision.models import inception_v3
            model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
            preprocess = T.Compose([
                T.Resize((299, 299)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            cls._instance = (model, preprocess)
            cls._device = device
        return cls._instance
    
    @staticmethod
    @torch.no_grad()
    def compute(pil_images, device="cuda", batch_size=32, splits=10):
        if not pil_images: return 0.0, 0.0
        model, preprocess = InceptionScoreMetric.get_instance(device)
        
        images = [Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img for img in pil_images]
        preds = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([preprocess(im) for im in images[i:i + batch_size]]).to(device)
            preds.append(F.softmax(model(x), dim=1).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        
        scores = []
        split_size = max(1, len(preds) // splits)
        for k in range(splits):
            part = preds[k * split_size:(k + 1) * split_size]
            if len(part) == 0: continue
            py = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
        
        return (float(np.mean(scores)), float(np.std(scores))) if scores else (0.0, 0.0)

# =========================
# I3D Model (Built-in from ProPainter)
# =========================
class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        return max(self.kernel_size[dim] - self.stride[dim], 0) if s % self.stride[dim] == 0 \
            else max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t, pad_h, pad_w = self.compute_pad(0, t), self.compute_pad(1, h), self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_t // 2, pad_t - pad_t // 2)
        return super().forward(F.pad(x, pad))

class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name='unit_3d'):
        super().__init__()
        self._output_channels, self._kernel_shape, self._stride = output_channels, kernel_shape, stride
        self._use_batch_norm, self._activation_fn, self._use_bias = use_batch_norm, activation_fn, use_bias
        self.name, self.padding = name, padding
        self.conv3d = nn.Conv3d(in_channels, output_channels, kernel_shape, stride, padding=0, bias=use_bias)
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        return max(self._kernel_shape[dim] - self._stride[dim], 0) if s % self._stride[dim] == 0 \
            else max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t, pad_h, pad_w = self.compute_pad(0, t), self.compute_pad(1, h), self.compute_pad(2, w)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_t // 2, pad_t - pad_t // 2)
        x = self.conv3d(F.pad(x, pad))
        if self._use_batch_norm: x = self.bn(x)
        return self._activation_fn(x) if self._activation_fn else x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], [1, 1, 1], name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels, out_channels[1], [1, 1, 1], name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(out_channels[1], out_channels[2], [3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels, out_channels[3], [1, 1, 1], name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(out_channels[3], out_channels[4], [3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels[5], [1, 1, 1], name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1b(self.b1a(x)), self.b2b(self.b2a(x)), self.b3b(self.b3a(x))], dim=1)

class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits',
                 name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown final endpoint {final_endpoint}')
        super().__init__()
        self._num_classes, self._spatial_squeeze, self._final_endpoint = num_classes, spatial_squeeze, final_endpoint
        self.logits, self.end_points = None, {}

        ep_configs = [
            ('Conv3d_1a_7x7', lambda: Unit3D(in_channels, 64, [7, 7, 7], (2, 2, 2), (3, 3, 3), name=name + 'Conv3d_1a_7x7')),
            ('MaxPool3d_2a_3x3', lambda: MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)),
            ('Conv3d_2b_1x1', lambda: Unit3D(64, 64, [1, 1, 1], name=name + 'Conv3d_2b_1x1')),
            ('Conv3d_2c_3x3', lambda: Unit3D(64, 192, [3, 3, 3], padding=1, name=name + 'Conv3d_2c_3x3')),
            ('MaxPool3d_3a_3x3', lambda: MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)),
            ('Mixed_3b', lambda: InceptionModule(192, [64, 96, 128, 16, 32, 32], name + 'Mixed_3b')),
            ('Mixed_3c', lambda: InceptionModule(256, [128, 128, 192, 32, 96, 64], name + 'Mixed_3c')),
            ('MaxPool3d_4a_3x3', lambda: MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)),
            ('Mixed_4b', lambda: InceptionModule(480, [192, 96, 208, 16, 48, 64], name + 'Mixed_4b')),
            ('Mixed_4c', lambda: InceptionModule(512, [160, 112, 224, 24, 64, 64], name + 'Mixed_4c')),
            ('Mixed_4d', lambda: InceptionModule(512, [128, 128, 256, 24, 64, 64], name + 'Mixed_4d')),
            ('Mixed_4e', lambda: InceptionModule(512, [112, 144, 288, 32, 64, 64], name + 'Mixed_4e')),
            ('Mixed_4f', lambda: InceptionModule(528, [256, 160, 320, 32, 128, 128], name + 'Mixed_4f')),
            ('MaxPool3d_5a_2x2', lambda: MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)),
            ('Mixed_5b', lambda: InceptionModule(832, [256, 160, 320, 32, 128, 128], name + 'Mixed_5b')),
            ('Mixed_5c', lambda: InceptionModule(832, [384, 192, 384, 48, 128, 128], name + 'Mixed_5c')),
        ]
        for ep_name, builder in ep_configs:
            self.end_points[ep_name] = builder()
            if self._final_endpoint == ep_name:
                self.build(); return

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(1024, self._num_classes, [1, 1, 1], activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.build()

    def build(self):
        for k in self.end_points: self.add_module(k, self.end_points[k])

    def forward(self, x):
        for ep in self.VALID_ENDPOINTS:
            if ep in self.end_points: x = self._modules[ep](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        return x.squeeze(3).squeeze(3) if self._spatial_squeeze else x

    def extract_features(self, x, target_endpoint='Logits'):
        for ep in self.VALID_ENDPOINTS:
            if ep in self.end_points:
                x = self._modules[ep](x)
                if ep == target_endpoint: break
        return x.mean(4).mean(3).mean(2) if target_endpoint == 'Logits' else x

# =========================
# ProPainter-official VFID (aligned)
# =========================
class _PP_Stack:
    def __init__(self, roll=False): self.roll = roll
    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1': img_group, mode = [img.convert('L') for img in img_group], 'L'
        if mode == 'L': return np.stack([np.expand_dims(np.array(x), 2) for x in img_group], axis=2)
        if mode == 'RGB': return np.stack([np.array(x)[:, :, ::-1] if self.roll else np.array(x) for x in img_group], axis=2)
        raise NotImplementedError(f"Image mode {mode}")

class _PP_ToTorchFormatTensor:
    def __init__(self, div=True): self.div = div
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode)).transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

def to_tensors():
    return T.Compose([_PP_Stack(), _PP_ToTorchFormatTensor()])

def init_i3d_model(i3d_model_path, device=None):
    print(f"[I3D] Loading from {i3d_model_path}")
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_path))
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return i3d_model.to(device)

def get_i3d_activations(batched_video, i3d_model, target_endpoint='Logits', flatten=True, grad_enabled=False):
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    return feat.view(feat.size(0), -1) if flatten else feat

def calculate_i3d_activations(video1, video2, i3d_model, device):
    video1 = to_tensors()(video1).unsqueeze(0).to(device)
    video2 = to_tensors()(video2).unsqueeze(0).to(device)
    return get_i3d_activations(video1, i3d_model).cpu().numpy().flatten(), \
           get_i3d_activations(video2, i3d_model).cpu().numpy().flatten()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def calculate_vfid(real_activations, fake_activations):
    m1, m2 = np.mean(real_activations, axis=0), np.mean(fake_activations, axis=0)
    s1, s2 = np.cov(real_activations, rowvar=False), np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)

# =========================
# E*_warp Metric (with RAFT optical flow)
# =========================
class RAFTFlowComputer:
    """RAFT-based optical flow computation."""
    _instance = None
    _device = None
    _raft_model_path = None
    
    @classmethod
    def get_instance(cls, device="cuda", raft_model_path=None):
        if cls._instance is None or cls._device != device or cls._raft_model_path != raft_model_path:
            if raft_model_path and os.path.exists(raft_model_path):
                try:
                    import sys
                    # Try different possible RAFT locations
                    possible_paths = [
                        "/home/hj/DiffuEraser/propainter",      # Current project
                        "/home/hj/DiffuEraser1/propainter",     # Alternative project
                        os.path.dirname(os.path.dirname(raft_model_path)),  # Infer from weights path
                    ]
                    
                    raft_module = None
                    for base_path in possible_paths:
                        raft_dir = os.path.join(base_path, "RAFT")
                        if os.path.exists(os.path.join(raft_dir, "raft.py")):
                            if base_path not in sys.path:
                                sys.path.insert(0, base_path)
                            try:
                                from RAFT.raft import RAFT
                                raft_module = RAFT
                                print(f"[RAFT] Found module at {raft_dir}")
                                break
                            except ImportError:
                                continue
                    
                    if raft_module is None:
                        # Try importing from propainter package
                        try:
                            from propainter.RAFT.raft import RAFT
                            raft_module = RAFT
                            print("[RAFT] Imported from propainter.RAFT")
                        except ImportError:
                            pass
                    
                    if raft_module is None:
                        raise ImportError("Could not find RAFT module in any known location")
                    
                    from argparse import Namespace
                    args = Namespace(small=False, mixed_precision=False)
                    model = raft_module(args)
                    
                    state_dict = torch.load(raft_model_path, map_location=device)
                    # Handle DataParallel weights
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace('module.', '') if k.startswith('module.') else k
                        new_state_dict[new_key] = v
                    model.load_state_dict(new_state_dict)
                    model = model.to(device).eval()
                    
                    cls._instance = model
                    cls._device = device
                    cls._raft_model_path = raft_model_path
                    print(f"[RAFT] Loaded weights from {raft_model_path}")
                except Exception as e:
                    print(f"[RAFT] Failed to load: {e}")
                    cls._instance = None
            else:
                print(f"[RAFT] Model not found at {raft_model_path}")
                cls._instance = None
        return cls._instance
    
    @staticmethod
    @torch.no_grad()
    def compute_flow(model, img1, img2, device="cuda", iters=20):
        """Compute optical flow from img1 to img2 using RAFT.
        
        Args:
            img1, img2: numpy arrays (H, W, 3) in RGB, uint8 [0-255]
        Returns:
            flow: numpy array (H, W, 2)
        """
        if model is None:
            return None
            
        H, W = img1.shape[:2]
        
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        # Convert to tensor
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Pad
        if pad_h > 0 or pad_w > 0:
            img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h), mode='replicate')
            img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h), mode='replicate')
        
        # Compute flow
        _, flow = model(img1_t, img2_t, iters=iters, test_mode=True)
        
        # Remove padding and convert to numpy
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        if pad_h > 0 or pad_w > 0:
            flow = flow[:H, :W, :]
        
        return flow.astype(np.float32)


class EwarpMetric:
    def __init__(self, device='cuda', raft_model_path=None, use_occlusion=True, preset='medium'):
        self.device = device
        self.use_occlusion = use_occlusion
        self.raft_model_path = raft_model_path
        self._raft_model = None
        self._use_raft = False
        
        # Try to load RAFT model
        if raft_model_path and os.path.exists(raft_model_path):
            self._raft_model = RAFTFlowComputer.get_instance(device, raft_model_path)
            if self._raft_model is not None:
                self._use_raft = True
                print("[Ewarp] Using RAFT for optical flow")
        
        # Fallback to DIS if RAFT not available
        if not self._use_raft:
            print("[Ewarp] Falling back to OpenCV DIS optical flow")
            preset_map = {'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                          'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST, 
                          'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
            self._dis = cv2.DISOpticalFlow_create(preset_map.get(preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
            try:
                self._dis.setFinestScale(1)
                self._dis.setPatchSize(8)
                self._dis.setPatchStride(4)
                self._dis.setGradientDescentIterations(25)
            except:
                pass

    @staticmethod
    def _to_float01_rgb(u8_rgb):
        return (u8_rgb if u8_rgb.dtype == np.uint8 else u8_rgb.astype(np.uint8)).astype(np.float32) / 255.0

    def _compute_flow(self, src_u8_rgb, dst_u8_rgb):
        """Compute optical flow from src to dst."""
        if self._use_raft:
            return RAFTFlowComputer.compute_flow(self._raft_model, src_u8_rgb, dst_u8_rgb, self.device)
        else:
            # Fallback to DIS
            src = self._to_float01_rgb(src_u8_rgb)
            dst = self._to_float01_rgb(dst_u8_rgb)
            src_gray = (cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            dst_gray = (cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            return self._dis.calc(src_gray, dst_gray, None).astype(np.float32)

    @staticmethod
    def _remap_img(img, map_x, map_y):
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def _remap_flow(flow, map_x, map_y):
        return np.stack([cv2.remap(flow[..., i], map_x, map_y, cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) 
                        for i in range(2)], axis=-1)

    def compute(self, out_frames_u8_rgb, masks01=None, gt_frames_u8_rgb=None, only_mask_region=False, scale=1000.0):
        n = len(out_frames_u8_rgb)
        if n < 2:
            return 0.0
        
        flow_src = gt_frames_u8_rgb if gt_frames_u8_rgb and len(gt_frames_u8_rgb) == n else out_frames_u8_rgb
        use_mask = only_mask_region and masks01 and len(masks01) == n
        errs = []

        for t in range(n - 1):
            try:
                B = self._compute_flow(flow_src[t + 1], flow_src[t])
                if B is None:
                    continue
                F = self._compute_flow(flow_src[t], flow_src[t + 1]) if self.use_occlusion else None
                
                H, W = B.shape[:2]
                xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                map_x, map_y = xs + B[..., 0], ys + B[..., 1]
                
                valid = (map_x >= 0) & (map_x <= W - 1) & (map_y >= 0) & (map_y <= H - 1)
                if self.use_occlusion and F is not None:
                    F_at = self._remap_flow(F, map_x, map_y)
                    fb = B + F_at
                    thr = 0.01 * (B[..., 0]**2 + B[..., 1]**2 + F_at[..., 0]**2 + F_at[..., 1]**2) + 0.5
                    valid &= (fb[..., 0]**2 + fb[..., 1]**2) <= thr
                if use_mask:
                    m = masks01[t + 1]
                    valid &= (m > 0.5) if m.dtype != np.bool_ else m
                if not valid.any():
                    continue

                warp_out = self._remap_img(self._to_float01_rgb(out_frames_u8_rgb[t]), map_x, map_y)
                diff = self._to_float01_rgb(out_frames_u8_rgb[t + 1]) - warp_out
                errs.append(float((diff**2).sum(axis=2)[valid].mean()))
            except Exception as e:
                print(f"[Ewarp] Frame {t} failed: {e}")
                continue

        return float(np.mean(errs) * scale) if errs else 0.0

# =========================
# Main Metrics Calculator
# =========================
class MetricsCalculator:
    def __init__(self, device="cuda", 
                 i3d_model_path="/home/hj/DiffuEraser1/weights/i3d_rgb_imagenet.pt",
                 aesthetic_ckpt_path="/home/hj/DiffuEraser1/weights/metrics/sa_0_4_vit_l_14_linear.pth",
                 raft_model_path="/home/hj/DiffuEraser1/weights/propainter/raft-things.pth",
                 **kwargs):
        self.device = device
        self.i3d_model_path = i3d_model_path
        self.aesthetic_ckpt_path = aesthetic_ckpt_path
        self.raft_model_path = raft_model_path
        self._i3d_model = None
        self._ewarp_metric = None
        
        # Handle legacy propainter_model_dir argument
        if 'propainter_model_dir' in kwargs and raft_model_path == "/home/hj/DiffuEraser1/weights/propainter/raft-things.pth":
            propainter_dir = kwargs['propainter_model_dir']
            potential_raft_path = os.path.join(propainter_dir, "raft-things.pth")
            if os.path.exists(potential_raft_path):
                self.raft_model_path = potential_raft_path
        
        print("[Metrics] Initializing models...")
        LPIPSMetric.get_instance(device)
        AestheticScoreMetric.get_instance(device, aesthetic_ckpt_path)
        InceptionScoreMetric.get_instance(device)
        
        if os.path.exists(i3d_model_path):
            self._i3d_model = init_i3d_model(i3d_model_path, device)
        else:
            print(f"[Metrics] Warning: I3D model not found at {i3d_model_path}")
        
        # Initialize Ewarp metric with RAFT
        if self.raft_model_path and os.path.exists(self.raft_model_path):
            self._ewarp_metric = EwarpMetric(device=device, raft_model_path=self.raft_model_path)
        else:
            print(f"[Metrics] Warning: RAFT model not found at {self.raft_model_path}, Ewarp will use DIS fallback")
            self._ewarp_metric = EwarpMetric(device=device, raft_model_path=None)
        
        print("[Metrics] Ready.")
    
    def compute_video_metrics(self, comp_frames, ori_frames, masks=None, compute_vfid=True, compute_is=True, compute_as=True, compute_ewarp=True):
        n_frames = min(len(comp_frames), len(ori_frames))
        comp_frames, ori_frames = comp_frames[:n_frames], ori_frames[:n_frames]
        if masks: masks = masks[:n_frames]
        
        psnr_list, ssim_list, lpips_list, as_list = [], [], [], []
        comp_pil_list, ori_pil_list = [], []
        
        for i in range(n_frames):
            comp, ori = comp_frames[i], ori_frames[i]
            if comp.shape[:2] != ori.shape[:2]:
                raise ValueError(f"Size mismatch: comp {comp.shape[:2]} vs ori {ori.shape[:2]}")
            
            psnr, ssim = calc_psnr_and_ssim(ori, comp)
            psnr_list.append(psnr); ssim_list.append(ssim)
            lpips_list.append(LPIPSMetric.compute(ori, comp, self.device))
            if compute_as:
                as_list.append(AestheticScoreMetric.compute(comp, self.device, self.aesthetic_ckpt_path))
            comp_pil_list.append(Image.fromarray(comp.astype(np.uint8)))
            ori_pil_list.append(Image.fromarray(ori.astype(np.uint8)))
        
        valid_psnr = [p for p in psnr_list if not np.isinf(p)]
        results = {
            'psnr_mean': np.mean(valid_psnr) if valid_psnr else 0.0,
            'ssim_mean': np.mean(ssim_list),
            'lpips_mean': np.mean(lpips_list),
        }
        if compute_as and as_list: results['as_mean'] = np.mean(as_list)
        if compute_is:
            is_mean, is_std = InceptionScoreMetric.compute(comp_pil_list, self.device)
            results['is_mean'], results['is_std'] = is_mean, is_std
        
        # Compute Ewarp
        if compute_ewarp and self._ewarp_metric:
            try:
                # Convert masks to binary if needed
                masks01 = None
                if masks:
                    masks01 = [(m > 127).astype(np.uint8) if m.max() > 1 else m for m in masks]
                ewarp_val = self._ewarp_metric.compute(comp_frames, masks01=masks01, gt_frames_u8_rgb=ori_frames)
                results['ewarp'] = ewarp_val
            except Exception as e:
                print(f"[Metrics] Ewarp failed: {e}")
                results['ewarp'] = -1.0
        
        if compute_vfid and self._i3d_model:
            try:
                ori_act, comp_act = calculate_i3d_activations(ori_pil_list, comp_pil_list, self._i3d_model, self.device)
                results['vfid'], results['ori_i3d_act'], results['comp_i3d_act'] = -1.0, ori_act, comp_act
            except Exception as e:
                print(f"[Metrics] VFID failed: {e}")
                results['vfid'] = -1.0
        else:
            results['vfid'] = -1.0
        return results
    
    def compute_final_vfid(self, all_ori_activations, all_comp_activations):
        if not all_ori_activations or not all_comp_activations: return -1.0
        try:
            return calculate_vfid(np.vstack(all_ori_activations), np.vstack(all_comp_activations))
        except Exception as e:
            print(f"[Metrics] Final VFID failed: {e}")
            return -1.0

# =========================
# Video I/O Helpers
# =========================
def read_video_frames(video_path, max_frames=None):
    if not video_path or not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames): break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames

def read_frame_sequence(frames_dir, max_frames=None):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if max_frames: frame_files = frame_files[:max_frames]
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frames_dir, f))
        if img: frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames

def read_mask_frames(mask_path, max_frames=None):
    if not mask_path or not os.path.exists(mask_path): return []
    cap = cv2.VideoCapture(mask_path)
    if not cap.isOpened(): return []
    masks, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames): break
        masks.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8))
        idx += 1
    cap.release()
    return masks
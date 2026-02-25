"""
三区 Mask 分解工具
将原始二值 mask 分解为 M_h (hole), M_b (boundary), M_c (context) 三个互斥区域。

用于 Region-Reg-DPO Loss 的区域加权计算。

约定:
  - 输入 mask: 1=hole, 0=valid（与训练 pipeline 中 mask 反转后一致）
  - 输出 M_h + M_b + M_c = 1（互斥且完备）
"""

import torch
import torch.nn.functional as F


def decompose_mask_regions(
    mask: torch.Tensor,
    dilation_kernel: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将二值 mask 分解为三个互斥区域。

    Args:
        mask: [B, 1, H, W] 或 [1, H, W]，值域 {0, 1}，1=hole
        dilation_kernel: 膨胀核大小（奇数），控制边界带宽度。
            - pixel space (512x512): 建议 7~15
            - latent space (64x64):  建议 3~5

    Returns:
        M_h: hole mask, shape 同 mask
        M_b: boundary mask, shape 同 mask
        M_c: context mask, shape 同 mask
        保证 M_h + M_b + M_c = 1
    """
    squeeze = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        squeeze = True

    # 确保二值化
    M_h = (mask > 0.5).float()

    # 膨胀操作：用 max pooling 模拟
    pad = dilation_kernel // 2
    dilated = F.max_pool2d(M_h, kernel_size=dilation_kernel, stride=1, padding=pad)

    # 边界带 = 膨胀区域 - 原始 hole
    M_b = (dilated - M_h).clamp(0, 1)

    # 上下文 = 剩余区域
    M_c = (1.0 - M_h - M_b).clamp(0, 1)

    if squeeze:
        M_h = M_h.squeeze(0)
        M_b = M_b.squeeze(0)
        M_c = M_c.squeeze(0)

    return M_h, M_b, M_c


def decompose_mask_regions_latent(
    mask: torch.Tensor,
    dilation_kernel_pixel: int = 7,
    latent_scale: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 latent space 下做三区分解。

    先将 mask downsample 到 latent 分辨率 (1/8)，再做分解。
    膨胀核自动缩放。

    Args:
        mask: [B, 1, H, W]，pixel space 分辨率
        dilation_kernel_pixel: pixel space 的膨胀核大小
        latent_scale: VAE 下采样倍数，默认 8

    Returns:
        M_h, M_b, M_c: latent space 分辨率的三区 mask
    """
    # downsample to latent space
    latent_mask = F.interpolate(
        mask, scale_factor=1.0 / latent_scale, mode="nearest"
    )

    # 缩放膨胀核（至少为 3）
    latent_kernel = max(3, dilation_kernel_pixel // latent_scale)
    # 确保奇数
    if latent_kernel % 2 == 0:
        latent_kernel += 1

    return decompose_mask_regions(latent_mask, dilation_kernel=latent_kernel)

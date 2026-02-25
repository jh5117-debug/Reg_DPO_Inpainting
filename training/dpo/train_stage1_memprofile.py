#!/usr/bin/env python
# coding=utf-8
"""
Stage 1 Memory Profiler — 不做实际训练，只跟踪每步 GPU 显存消耗。
用法:
  CUDA_VISIBLE_DEVICES=2 python train_stage1_memprofile.py \
    --base_model_name_or_path weights/stable-diffusion-v1-5 \
    --pretrained_stage1_path weights/diffuEraser \
    --vae_path weights/sd-vae-ft-mse \
    --resolution 512 --nframes 10 --mixed_precision bf16
"""

import argparse
import gc
import torch
from einops import rearrange, repeat
import torch.nn.functional as F

# ── helpers ──
def mem_mb():
    """Return (allocated_MB, reserved_MB) on current device."""
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    return a, r

def log_mem(tag):
    a, r = mem_mb()
    print(f"  [MEM] {tag:48s} | alloc {a:8.1f} MB | reserved {r:8.1f} MB")

def count_params(model, name="model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  [PARAMS] {name:20s} | total: {total/1e6:.1f}M | trainable: {trainable/1e6:.1f}M | size: {bytes_total/1024**2:.1f} MB")
    return total, trainable

# ── main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_stage1_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--nframes", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"Resolution: {args.resolution}, nframes: {args.nframes}, mixed_precision: {args.mixed_precision}")
    print(f"gradient_checkpointing: {args.gradient_checkpointing}")
    print("=" * 70)

    log_mem("0. Initial")

    # ────────────────────────────────────────────
    # 1. Load models
    # ────────────────────────────────────────────
    print("\n>>> Step 1: Loading models...")

    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    log_mem("1a. VAE loaded (frozen, half)")
    count_params(vae, "VAE")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    log_mem("1b. TextEncoder loaded (frozen, half)")
    count_params(text_encoder, "TextEncoder")

    from libs.unet_2d_condition import UNet2DConditionModel
    from libs.brushnet_CA import BrushNetModel

    if args.pretrained_stage1_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model_name_or_path, subfolder="unet"
        )
        brushnet = BrushNetModel.from_pretrained(
            args.pretrained_stage1_path, subfolder="brushnet"
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.base_model_name_or_path, subfolder="unet"
        )
        brushnet = BrushNetModel.from_unet(unet)

    unet.to(device)       # float32 for training
    brushnet.to(device)   # float32 for training
    unet.train()
    brushnet.train()
    log_mem("1c. UNet + BrushNet loaded (trainable, float32)")

    n_unet, t_unet = count_params(unet, "UNet2D")
    n_brush, t_brush = count_params(brushnet, "BrushNet")

    if args.gradient_checkpointing:
        try:
            unet.enable_gradient_checkpointing()
            print("  >> UNet gradient checkpointing ENABLED")
        except Exception as e:
            print(f"  >> UNet gradient checkpointing FAILED: {e}")
        try:
            brushnet.enable_gradient_checkpointing()
            print("  >> BrushNet gradient checkpointing ENABLED")
        except Exception as e:
            print(f"  >> BrushNet gradient checkpointing FAILED: {e}")


    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_name_or_path, subfolder="scheduler"
    )

    # ────────────────────────────────────────────
    # 2. Create optimizer
    # ────────────────────────────────────────────
    print("\n>>> Step 2: Creating AdamW optimizer...")
    params = list(unet.parameters()) + list(brushnet.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    adam_state_estimate = total_trainable * 4 * 2  # 2 states (m, v), each float32
    print(f"  Trainable params: {total_trainable/1e6:.1f}M")
    print(f"  Estimated Adam states: {adam_state_estimate/1024**2:.1f} MB")
    print(f"  Estimated gradients: {total_trainable * 4 / 1024**2:.1f} MB")

    optimizer = torch.optim.AdamW(trainable_params, lr=5e-6)
    log_mem("2. Optimizer created (states NOT yet allocated)")

    # ────────────────────────────────────────────
    # 3. Create dummy data
    # ────────────────────────────────────────────
    print(f"\n>>> Step 3: Creating dummy data (bs=1, nframes={args.nframes}, res={args.resolution})...")
    B = 1
    F_n = args.nframes
    H = W = args.resolution

    # Simulate dataset output
    pixel_values = torch.randn(B, F_n, 3, H, W, device=device, dtype=weight_dtype)
    cond_pixel_values = torch.randn(B, F_n, 3, H, W, device=device, dtype=weight_dtype)
    masks = torch.ones(B, F_n, 1, H, W, device=device, dtype=weight_dtype)
    input_ids = torch.randint(0, 49408, (B, 77), device=device)
    log_mem("3. Dummy data created")

    # ────────────────────────────────────────────
    # 4. Forward pass
    # ────────────────────────────────────────────
    print("\n>>> Step 4: Forward pass...")

    # 4a. VAE encode
    with torch.no_grad():
        latents = vae.encode(
            rearrange(pixel_values, "b f c h w -> (b f) c h w")
        ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    log_mem("4a. VAE encode (ground truth)")

    with torch.no_grad():
        cond_latents = vae.encode(
            rearrange(cond_pixel_values, "b f c h w -> (b f) c h w")
        ).latent_dist.sample()
        cond_latents = cond_latents * vae.config.scaling_factor
        cond_latents = rearrange(cond_latents, "(b f) c h w -> b f c h w", b=B)
    log_mem("4b. VAE encode (conditioning)")

    # Free pixel data
    del pixel_values, cond_pixel_values
    torch.cuda.empty_cache()

    # Mask interpolation
    masks_latent = torch.nn.functional.interpolate(
        masks.to(dtype=weight_dtype).view(B * F_n, 1, H, W),
        size=(latents.shape[-2], latents.shape[-1])
    ).view(B, F_n, 1, latents.shape[-2], latents.shape[-1])
    
    cond_latents = rearrange(
        torch.cat([cond_latents, masks_latent], 2),
        "b f c h w -> (b f) c h w"
    )
    log_mem("4c. Conditioning latents prepared")

    # Noise + timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (B,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    log_mem("4d. Noise added")

    # Text encoding
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
    log_mem("4e. Text encoded")

    # 4f. BrushNet forward
    print("  >> Running BrushNet forward...")
    
    # 根据参数选择 autocast dtype
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    # 如果 mixed_precision="no"，则不使用 autocast (dtype=None 或 float32)
    use_amp = args.mixed_precision in ["fp16", "bf16"]
    
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        down_block_res_samples, mid_block_res_sample, up_block_res_samples = brushnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=rearrange(
                repeat(encoder_hidden_states, "b c d -> b t c d", t=F_n),
                "b t c d -> (b t) c d"
            ),
            brushnet_cond=cond_latents,
            return_dict=False,
        )
    log_mem("4f. BrushNet forward done")

    torch.cuda.empty_cache()
    gc.collect()

    # 4g. UNet forward
    print("  >> Running UNet forward...")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=rearrange(
                repeat(encoder_hidden_states, "b c d -> b t c d", t=F_n),
                "b t c d -> (b t) c d"
            ),
            down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_block_res_samples],
            mid_block_add_sample=mid_block_res_sample.to(dtype=weight_dtype),
            up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_block_res_samples],
            return_dict=True,
        ).sample

    log_mem("4g. UNet forward done")

    torch.cuda.empty_cache()
    gc.collect()

    # ────────────────────────────────────────────
    # 5. Loss + Backward
    # ────────────────────────────────────────────
    print("\n>>> Step 5: Loss + Backward...")
    target = noise
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    log_mem("5a. Loss computed")

    loss.backward()
    log_mem("5b. Backward done (gradients allocated)")

    torch.cuda.empty_cache()
    gc.collect()

    # ────────────────────────────────────────────
    # 6. Optimizer step (THIS IS WHERE OOM HAPPENS)
    # ────────────────────────────────────────────
    print("\n>>> Step 6: optimizer.step() — Adam states will be allocated here...")
    print(f"  ⚠️  Expected additional VRAM: ~{adam_state_estimate/1024**2:.0f} MB")
    a_before, _ = mem_mb()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
    remaining = total_vram - a_before
    print(f"  Current allocated: {a_before:.0f} MB, remaining: {remaining:.0f} MB")
    if remaining < adam_state_estimate / 1024**2:
        print(f"  ❌ WILL OOM! Need {adam_state_estimate/1024**2:.0f} MB but only {remaining:.0f} MB left")
        print(f"  ❌ Deficit: {adam_state_estimate/1024**2 - remaining:.0f} MB")
    else:
        print(f"  ✅ Should fit. Headroom: {remaining - adam_state_estimate/1024**2:.0f} MB")

    try:
        optimizer.step()
        log_mem("6. optimizer.step() SUCCEEDED")
    except torch.cuda.OutOfMemoryError:
        log_mem("6. optimizer.step() OOM! <<<")
        print("\n" + "=" * 70)
        print("❌ OOM at optimizer.step() — Adam states exceeded VRAM")
        print("=" * 70)

    # ────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MEMORY SUMMARY")
    print("=" * 70)
    a, r = mem_mb()
    print(f"  Final allocated: {a:.1f} MB")
    print(f"  Final reserved:  {r:.1f} MB")
    print(f"  Total VRAM:      {total_vram:.0f} MB")
    print()
    print("BREAKDOWN ESTIMATE (float32 training):")
    print(f"  UNet params:           {n_unet * 4 / 1024**2:.1f} MB")
    print(f"  BrushNet params:       {n_brush * 4 / 1024**2:.1f} MB")
    print(f"  Gradients:             {(t_unet + t_brush) * 4 / 1024**2:.1f} MB")
    print(f"  Adam exp_avg:          {(t_unet + t_brush) * 4 / 1024**2:.1f} MB")
    print(f"  Adam exp_avg_sq:       {(t_unet + t_brush) * 4 / 1024**2:.1f} MB")
    print(f"  ─────────────────────────────────────")
    total_model = (n_unet + n_brush) * 4 + (t_unet + t_brush) * 4 * 3  # params + grads + 2 adam states
    print(f"  Subtotal (model+opt):  {total_model / 1024**2:.1f} MB")
    print(f"  VAE + TextEncoder:     ~800 MB (half precision)")
    print(f"  Activations:           dynamic (depends on resolution & nframes)")
    print()

    # ────────────────────────────────────────────
    # Suggestions
    # ────────────────────────────────────────────
    print("POSSIBLE SOLUTIONS:")
    deficit = total_model / 1024**2 + 800 - total_vram
    if deficit > 0:
        print(f"  ⚠️  Model+optimizer alone exceeds VRAM by ~{deficit:.0f} MB")
        print()
        
    print("  1. [推荐] 冻结 UNet, 只训 BrushNet:")
    brushnet_only = n_brush * 4 + t_brush * 4 * 3
    print(f"     → 优化器显存: {brushnet_only/1024**2:.1f} MB (节省 {(total_model - brushnet_only - n_unet*4)/1024**2:.0f} MB)")
    
    print("  2. [推荐] 使用 8-bit Adam (bitsandbytes):")
    adam_8bit = (t_unet + t_brush) * 1 * 2  # 1 byte each
    saved = adam_state_estimate - adam_8bit
    print(f"     → Adam states: {adam_8bit/1024**2:.1f} MB (节省 {saved/1024**2:.0f} MB)")
    
    print("  3. 冻结 UNet + 8-bit Adam:")
    combo = n_unet * 4 + n_brush * 4 + t_brush * 4 + t_brush * 1 * 2
    print(f"     → 总模型+优化器: {combo/1024**2:.1f} MB")


if __name__ == "__main__":
    main()

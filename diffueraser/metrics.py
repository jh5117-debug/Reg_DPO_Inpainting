
# =====================================================================================
# ProPainter-like E*_warp override
# - Flow computed on GT frames (if provided).
# - Warp error uses MSE on non-occluded/in-bounds pixels.
# - Return value is Ewarp * scale, default scale=1000 to match paper's E*_warp (×10^-3).
# =====================================================================================
import numpy as _np
import cv2 as _cv2

class EwarpMetric:
    def __init__(self, device='cuda', preset='medium', use_occlusion=True):
        self.device = device
        self.use_occlusion = use_occlusion
        preset_map = {
            'ultrafast': _cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            'fast': _cv2.DISOPTICAL_FLOW_PRESET_FAST,
            'medium': _cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        self._dis = _cv2.DISOpticalFlow_create(preset_map.get(preset, _cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
        # relatively stable DIS settings
        try:
            self._dis.setFinestScale(1)
            self._dis.setPatchSize(8)
            self._dis.setPatchStride(4)
            self._dis.setGradientDescentIterations(25)
        except Exception:
            pass

    @staticmethod
    def _to_float01_rgb(u8_rgb):
        if u8_rgb.dtype != _np.uint8:
            u8_rgb = u8_rgb.astype(_np.uint8)
        return u8_rgb.astype(_np.float32) / 255.0

    def _flow_dis(self, src_u8_rgb, dst_u8_rgb):
        # OpenCV DIS: flow on src grid, mapping src -> dst
        src = self._to_float01_rgb(src_u8_rgb)
        dst = self._to_float01_rgb(dst_u8_rgb)
        src_g = _cv2.cvtColor(src, _cv2.COLOR_RGB2GRAY)
        dst_g = _cv2.cvtColor(dst, _cv2.COLOR_RGB2GRAY)
        return self._dis.calc(src_g, dst_g, None).astype(_np.float32)  # H,W,2

    @staticmethod
    def _remap_img(img_f01_rgb, map_x, map_y):
        return _cv2.remap(
            img_f01_rgb, map_x, map_y,
            interpolation=_cv2.INTER_LINEAR,
            borderMode=_cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _remap_flow(flow, map_x, map_y):
        fx = _cv2.remap(flow[..., 0], map_x, map_y, interpolation=_cv2.INTER_LINEAR,
                        borderMode=_cv2.BORDER_CONSTANT, borderValue=0).astype(_np.float32)
        fy = _cv2.remap(flow[..., 1], map_x, map_y, interpolation=_cv2.INTER_LINEAR,
                        borderMode=_cv2.BORDER_CONSTANT, borderValue=0).astype(_np.float32)
        return _np.stack([fx, fy], axis=-1)

    def compute(self, out_frames_u8_rgb, masks01=None, gt_frames_u8_rgb=None,
                only_mask_region=False, scale=1000.0):
        """
        Return Ewarp * scale (default 1000, matching paper's E*_warp column).

        out_frames_u8_rgb: list of uint8 RGB frames (method output)
        gt_frames_u8_rgb:  list of uint8 RGB frames (GT/input). If provided, flow is computed on GT.
        masks01:           list of HxW masks (1=hole). Used only if only_mask_region=True.
        """
        n = len(out_frames_u8_rgb)
        if n < 2:
            return 0.0

        if gt_frames_u8_rgb is not None and len(gt_frames_u8_rgb) == n:
            flow_src = gt_frames_u8_rgb
        else:
            flow_src = out_frames_u8_rgb

        use_mask = only_mask_region and (masks01 is not None) and (len(masks01) == n)
        errs = []

        for t in range(n - 1):
            src_t   = flow_src[t]
            src_tp1 = flow_src[t + 1]

            # Backward flow on (t+1) grid: (t+1)->t  (good for remap out_t into t+1 coords)
            B = self._flow_dis(src_tp1, src_t)      # H,W,2 on t+1 grid
            F = self._flow_dis(src_t, src_tp1) if self.use_occlusion else None  # H,W,2 on t grid

            H, W = B.shape[:2]
            xs, ys = _np.meshgrid(_np.arange(W, dtype=_np.float32),
                                  _np.arange(H, dtype=_np.float32))

            # map from (t+1) pixel -> (t) sampling location
            map_x = xs + B[..., 0]
            map_y = ys + B[..., 1]

            inb = (map_x >= 0) & (map_x <= (W - 1)) & (map_y >= 0) & (map_y <= (H - 1))
            valid = inb

            if self.use_occlusion and F is not None:
                # sample forward flow at mapped coords (now on t+1 grid)
                F_at = self._remap_flow(F, map_x.astype(_np.float32), map_y.astype(_np.float32))
                fb = B + F_at
                fb2 = fb[..., 0]**2 + fb[..., 1]**2
                B2  = B[..., 0]**2  + B[..., 1]**2
                F2  = F_at[..., 0]**2 + F_at[..., 1]**2
                thr = 0.01 * (B2 + F2) + 0.5
                noc = fb2 <= thr
                valid = valid & noc

            if use_mask:
                m = masks01[t + 1]
                if m.dtype != _np.bool_:
                    m = (m > 0.5)
                valid = valid & m

            if not valid.any():
                continue

            out_t   = self._to_float01_rgb(out_frames_u8_rgb[t])
            out_tp1 = self._to_float01_rgb(out_frames_u8_rgb[t + 1])

            # warp out_t into (t+1) coords using backward flow B
            warp_out_t = self._remap_img(out_t, map_x.astype(_np.float32), map_y.astype(_np.float32))

            diff = out_tp1 - warp_out_t
            diff2 = (diff**2).sum(axis=2)  # RGB channel sum
            errs.append(float(diff2[valid].mean()))

        if not errs:
            return 0.0
        return float(_np.mean(errs) * float(scale))

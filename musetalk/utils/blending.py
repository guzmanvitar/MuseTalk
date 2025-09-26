from PIL import Image
import numpy as np
import cv2
import copy
from collections import deque
from typing import Tuple


# Optional foreground (hands) detection for preservation during blending
try:
    from musetalk.utils.foreground import detect_hands_mask
except Exception:  # optional dependency; feature gated by CLI flag
    detect_hands_mask = None

# One-time info log toggle for foreground preservation
_FGP_LOG_ONCE = False

# Temporal smoothing state (single-stream per process)
_FGP_SMOOTH_Q: deque[np.ndarray] = deque(maxlen=0)
_FGP_SMOOTH_WIN: int = 0
_FGP_SMOOTH_SHAPE: tuple[int, int] | None = None


def reset_foreground_preserve_state(window: int | None = None):
    """Reset temporal smoothing buffer between clips.
    Optionally set a new smoothing window size (0 disables).
    """
    global _FGP_SMOOTH_Q, _FGP_SMOOTH_WIN, _FGP_SMOOTH_SHAPE
    _FGP_SMOOTH_Q.clear()
    _FGP_SMOOTH_SHAPE = None
    if window is not None:
        _FGP_SMOOTH_WIN = max(0, int(window))
        _FGP_SMOOTH_Q = deque(maxlen=_FGP_SMOOTH_WIN if _FGP_SMOOTH_WIN > 0 else 0)


def _smooth_mask(mask: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return mask
    global _FGP_SMOOTH_Q, _FGP_SMOOTH_SHAPE
    if _FGP_SMOOTH_Q.maxlen != window:
        _FGP_SMOOTH_Q = deque(maxlen=window)
        _FGP_SMOOTH_SHAPE = None
    # If shape changes (crop size varies), reset buffer to avoid size mismatch
    if _FGP_SMOOTH_SHAPE is None or _FGP_SMOOTH_SHAPE != mask.shape:
        _FGP_SMOOTH_Q.clear()
        _FGP_SMOOTH_SHAPE = mask.shape
    _FGP_SMOOTH_Q.append(mask)
    # Bitwise OR over window (conservative preserve)
    out = _FGP_SMOOTH_Q[0].copy()
    for m in list(_FGP_SMOOTH_Q)[1:]:
        out = cv2.bitwise_or(out, m)
    return out


def _scale_box_centered(box: Tuple[int,int,int,int], scale: float, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0) * scale
    h = (y1 - y0) * scale
    nx0 = int(round(cx - w / 2.0))
    ny0 = int(round(cy - h / 2.0))
    nx1 = int(round(cx + w / 2.0))
    ny1 = int(round(cy + h / 2.0))
    nx0 = max(0, nx0); ny0 = max(0, ny0)
    nx1 = min(img_w, nx1); ny1 = min(img_h, ny1)
    return nx0, ny0, nx1, ny1


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(
    image,
    face,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    mode="raw",
    fp=None,
    enable_foreground_preserve: bool = False,
    foreground_preserve_dilate_px: int = 5,
    foreground_preserve_debug: bool = False,
    foreground_preserve_temporal_smooth: int = 0,
    foreground_preserve_roi_scale: float = 1.0,
    foreground_preserve_min_det_conf: float = 0.4,
    foreground_preserve_min_track_conf: float = 0.3,
):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
    ori_shape = face_large.size  # (W, H)

    if foreground_preserve_debug:
        print(f"[FGP] get_image() entry; enable={enable_foreground_preserve}; dilate_px={foreground_preserve_dilate_px}")

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码

    full_mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    full_mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上

    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = full_mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(full_mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码

    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊

    # 可选：前景（手部）区域保护，避免覆盖与羽化过渡导致的手部边缘模糊
    global _FGP_LOG_ONCE
    if enable_foreground_preserve and not _FGP_LOG_ONCE:
        print(f"[FGP] enabled=True; mp_available={detect_hands_mask is not None}; debug={foreground_preserve_debug}; "
              f"dilate_px={foreground_preserve_dilate_px}; roi_scale={foreground_preserve_roi_scale}; "
              f"temporal_win={foreground_preserve_temporal_smooth}; min_det={foreground_preserve_min_det_conf}; min_track={foreground_preserve_min_track_conf}")
        _FGP_LOG_ONCE = True

    did_preserve = False
    if enable_foreground_preserve:
        if detect_hands_mask is None:
            if foreground_preserve_debug:
                print("[FGP] detect_hands_mask unavailable; skipping preserve")
        else:
            try:
                # Build a larger detection ROI around the current crop to reduce dropouts
                img_w, img_h = body.size
                dx0, dy0, dx1, dy1 = _scale_box_centered(crop_box, max(1.0, float(foreground_preserve_roi_scale)), img_w, img_h)
                detect_region = body.crop((dx0, dy0, dx1, dy1))
                detect_bgr = np.array(detect_region)[:, :, ::-1]

                preserve_detect = detect_hands_mask(
                    detect_bgr,
                    min_det_conf=float(foreground_preserve_min_det_conf),
                    min_track_conf=float(foreground_preserve_min_track_conf),
                )
                if preserve_detect is None:
                    if foreground_preserve_debug:
                        print("[FGP] mediapipe returned None preserve mask")
                    preserve_detect = np.zeros((detect_bgr.shape[0], detect_bgr.shape[1]), dtype=np.uint8)

                # Dilate for safety margin around edges
                k = max(1, int(foreground_preserve_dilate_px))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                preserve_detect = cv2.dilate(preserve_detect, kernel, iterations=1)

                # Map detection mask back onto face_large coordinates
                ori_w, ori_h = ori_shape[0], ori_shape[1]
                preserve_large = np.zeros((ori_h, ori_w), dtype=np.uint8)
                inter_left = max(dx0, x_s)
                inter_top = max(dy0, y_s)
                inter_right = min(dx1, x_e)
                inter_bottom = min(dy1, y_e)
                if inter_right > inter_left and inter_bottom > inter_top:
                    w_int = inter_right - inter_left
                    h_int = inter_bottom - inter_top
                    dx = inter_left - dx0
                    dy = inter_top - dy0
                    fx = inter_left - x_s
                    fy = inter_top - y_s
                    preserve_large[fy:fy+h_int, fx:fx+w_int] = preserve_detect[dy:dy+h_int, dx:dx+w_int]

                # Temporal smoothing (bitwise OR over last N masks)
                preserve_large = _smooth_mask(preserve_large, int(foreground_preserve_temporal_smooth))

                # Compute small-region trimmed mask used for pasting the face crop
                x0, y0, x1o, y1o = (x - x_s), (y - y_s), (x1 - x_s), (y1 - y_s)
                preserve_small = preserve_large[y0:y1o, x0:x1o]
                mask_small_np = np.array(mask_small)
                face_small_mask_trim = ((mask_small_np > 0) & (preserve_small == 0)).astype(np.uint8) * 255

                # 用修剪后的mask粘贴面部区域，避免覆盖手部像素
                face_large.paste(face, (x0, y0, x1o, y1o), Image.fromarray(face_small_mask_trim))

                # 从最终羽化后的mask中减去扩张后的前景保留区域，避免手部边缘出现羽化混合
                mask_array_final = mask_array.copy()
                mask_array_final[preserve_large > 0] = 0
                mask_image_final = Image.fromarray(mask_array_final)
                did_preserve = True

                if foreground_preserve_debug:
                    nz = int((preserve_large > 0).sum())
                    print(f"[FGP] preserve applied; crop=({x_s},{y_s},{x_e},{y_e}) detect=({dx0},{dy0},{dx1},{dy1}) px={nz}")
            except Exception as e:
                if foreground_preserve_debug:
                    print(f"[FGP] error in preserve path: {e}")
                did_preserve = False

    if not did_preserve:
        # 默认逻辑：不使用保护，直接粘贴
        face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        mask_image_final = Image.fromarray(mask_array)

    # 合成回原图
    body.paste(face_large, crop_box[:2], mask_image_final)
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组
    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(
    image,
    face,
    face_box,
    mask_array,
    crop_box,
    enable_foreground_preserve: bool = False,
    foreground_preserve_dilate_px: int = 5,
    foreground_preserve_debug: bool = False,
    foreground_preserve_temporal_smooth: int = 0,
    foreground_preserve_roi_scale: float = 1.0,
    foreground_preserve_min_det_conf: float = 0.4,
    foreground_preserve_min_track_conf: float = 0.3,
):
    global _FGP_LOG_ONCE
    body = Image.fromarray(image[:, :, ::-1])
    face = Image.fromarray(face[:, :, ::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    if foreground_preserve_debug and not _FGP_LOG_ONCE:
        print(f"[FGP] (realtime) get_image_blending() entry; enable={enable_foreground_preserve}; mp_available={detect_hands_mask is not None}; "
              f"dilate_px={foreground_preserve_dilate_px}; roi_scale={foreground_preserve_roi_scale}; temporal_win={foreground_preserve_temporal_smooth}; "
              f"min_det={foreground_preserve_min_det_conf}; min_track={foreground_preserve_min_track_conf}")
        _FGP_LOG_ONCE = True
    elif enable_foreground_preserve and not _FGP_LOG_ONCE:
        print(f"[FGP] (realtime) enabled=True; mp_available={detect_hands_mask is not None}; debug={foreground_preserve_debug}; "
              f"dilate_px={foreground_preserve_dilate_px}; roi_scale={foreground_preserve_roi_scale}; temporal_win={foreground_preserve_temporal_smooth}; "
              f"min_det={foreground_preserve_min_det_conf}; min_track={foreground_preserve_min_track_conf}")
        _FGP_LOG_ONCE = True

    did_preserve = False
    if enable_foreground_preserve:
        if detect_hands_mask is None:
            if foreground_preserve_debug:
                print("[FGP] (realtime) detect_hands_mask unavailable; skipping preserve")
        else:
            try:
                # Larger detection ROI centered on the current crop
                img_w, img_h = body.size
                dx0, dy0, dx1, dy1 = _scale_box_centered(crop_box, max(1.0, float(foreground_preserve_roi_scale)), img_w, img_h)
                detect_region = body.crop((dx0, dy0, dx1, dy1))
                detect_bgr = np.array(detect_region)[:, :, ::-1]

                preserve_detect = detect_hands_mask(
                    detect_bgr,
                    min_det_conf=float(foreground_preserve_min_det_conf),
                    min_track_conf=float(foreground_preserve_min_track_conf),
                )
                if preserve_detect is None:
                    if foreground_preserve_debug:
                        print("[FGP] (realtime) mediapipe returned None preserve mask")
                    preserve_detect = np.zeros((detect_bgr.shape[0], detect_bgr.shape[1]), dtype=np.uint8)

                # Dilate for margin
                k = max(1, int(foreground_preserve_dilate_px))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                preserve_detect = cv2.dilate(preserve_detect, kernel, iterations=1)

                # Map to face_large coords
                ori_w, ori_h = face_large.size[0], face_large.size[1]
                preserve_large = np.zeros((ori_h, ori_w), dtype=np.uint8)
                inter_left = max(dx0, x_s)
                inter_top = max(dy0, y_s)
                inter_right = min(dx1, x_e)
                inter_bottom = min(dy1, y_e)
                if inter_right > inter_left and inter_bottom > inter_top:
                    w_int = inter_right - inter_left
                    h_int = inter_bottom - inter_top
                    dx = inter_left - dx0
                    dy = inter_top - dy0
                    fx = inter_left - x_s
                    fy = inter_top - y_s
                    preserve_large[fy:fy+h_int, fx:fx+w_int] = preserve_detect[dy:dy+h_int, dx:dx+w_int]

                # Temporal smoothing
                preserve_large = _smooth_mask(preserve_large, int(foreground_preserve_temporal_smooth))

                # Build a small paste mask from incoming mask_array, trimmed by preserve_small
                x0, y0, x1o, y1o = (x - x_s), (y - y_s), (x1 - x_s), (y1 - y_s)
                preserve_small = preserve_large[y0:y1o, x0:x1o]
                paste_small = ((mask_array[y0:y1o, x0:x1o] > 0) & (preserve_small == 0)).astype(np.uint8) * 255

                face_large.paste(face, (x0, y0, x1o, y1o), Image.fromarray(paste_small))

                mask_array_final = mask_array.copy()
                mask_array_final[preserve_large > 0] = 0
                mask_image = Image.fromarray(mask_array_final).convert("L")
                did_preserve = True

                if foreground_preserve_debug:
                    nz = int((preserve_large > 0).sum())
                    print(f"[FGP] (realtime) preserve applied; crop=({x_s},{y_s},{x_e},{y_e}) detect=({dx0},{dy0},{dx1},{dy1}) px={nz}")
            except Exception as e:
                if foreground_preserve_debug:
                    print(f"[FGP] (realtime) error in preserve path: {e}")
                did_preserve = False

    if not did_preserve:
        mask_image = Image.fromarray(mask_array).convert("L")
        face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:, :, ::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box

import numpy as np
import cv2
from typing import Optional

# NOTE: We keep all foreground (hands/objects) detection logic isolated here
# so it can be tested independently and extended later without touching
# core blending code.

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    mp = None


_HANDS_CTX = None
_HANDS_CONF = None  # track last init params to allow re-init when flags change

def _get_hands(min_det_conf: float = 0.4, min_track_conf: float = 0.3):
    """Lazy-init and return a global MediaPipe Hands context if available.

    Re-initializes when requested confidences differ from the current context.
    Returns None when MediaPipe is not installed; the caller should treat this
    as no-foreground detected and proceed without preservation.
    """
    global _HANDS_CTX, _HANDS_CONF
    if mp is None:
        return None
    req = {"min_det": float(min_det_conf), "min_track": float(min_track_conf)}
    if _HANDS_CTX is None or _HANDS_CONF != req:
        # static_image_mode=True is good for per-frame invocation
        try:
            if _HANDS_CTX is not None:
                _HANDS_CTX.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        _HANDS_CTX = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf),
        )
        _HANDS_CONF = req
    return _HANDS_CTX


def detect_hands_mask(
    image_bgr: np.ndarray,
    min_det_conf: float = 0.4,
    min_track_conf: float = 0.3,
) -> Optional[np.ndarray]:
    """Detect hands in the given BGR image and return a uint8 mask (0/255).

    - Input: image_bgr (H, W, 3), dtype=uint8
    - Output: mask (H, W), dtype=uint8, 255 where hands are detected, else 0
    - Returns None if mediapipe is not available or on internal failure

    The mask is intentionally a coarse convex hull around landmarks to be robust
    under motion blur. Callers should optionally dilate it via a small kernel
    (e.g., 3-7 px) before subtracting from blending masks.
    """
    try:
        h, w = image_bgr.shape[:2]
        hands = _get_hands(min_det_conf, min_track_conf)
        if hands is None:
            return None
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        if not getattr(result, "multi_hand_landmarks", None):
            return np.zeros((h, w), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        for hand_landmarks in result.multi_hand_landmarks:
            pts = []
            for lm in hand_landmarks.landmark:
                x = int(round(lm.x * w))
                y = int(round(lm.y * h))
                pts.append([x, y])
            if len(pts) >= 3:
                hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
        return mask
    except Exception:
        # Fail closed: no preserve mask
        return None


# ================================
# Future extensions (placeholders)
# ================================
# TODO [mic-seg]: Integrate microphone detection/segmentation (e.g., YOLOv8n-seg or
#                 a small custom model). Provide a detect_microphone_mask(image)
#                 API returning a uint8 mask aligned to the input image.
#
# TODO [generic-objects]: Add generic object detection for common props (phones,
#                 remotes, pens). Either via general-purpose instance segmentation
#                 (e.g., YOLOv8-seg on COCO classes) or a light custom model.
#                 Expose detect_generic_objects_mask(image) that unions all props.
#
# TODO [depth-aware]: Plug a lightweight monocular depth model (e.g., MiDaS small)
#                 to compute per-pixel depth for frames flagged with occlusion.
#                 Use it to refine occlusion ordering and preserve closer pixels:
#                 preserve_mask = (depth_foreground < depth_face_region).
#                 Keep this feature optional and gated behind CLI flags.


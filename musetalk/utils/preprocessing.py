import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm
from collections import deque

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient
coord_placeholder = (0.0,0.0,0.0,0.0)

# Global variables for pitch filtering
pitch_filter_enabled = False
pitch_down_threshold_deg = 60.0
pitch_up_threshold_deg = 55.0
pitch_history = deque(maxlen=5)  # Store last 5 smoothed pitch values
pitch_state = "normal"  # "normal" or "filtered"
consecutive_frames_threshold = 2
# Option 1 enhancements (confidence + EMA + min-hold + debug)
pitch_conf_min = 0.6
pitch_ema_alpha = 0.3
ema_pitch = None
pitch_min_hold_frames = 6
last_state_change_frame_index = -10**9
pitch_debug_detailed = False
# 2D pose filter globals (ratios + simple hysteresis)
pose2d_filter_enabled = False
pose2d_vpi_thr = 1.20
pose2d_lfc_thr = 0.20
pose2d_nmi_thr = 0.55
pose2d_ema_alpha = 0.30
pose2d_consec_frames = 3
pose2d_min_hold_frames = 6
pose2d_none_consecutive_max = 4
pose2d_enable_ear_gate = False
pose2d_ear_thr = 0.18
pose2d_ear_gate_consec = 2
pose2d_ear_bias = 0.02
pose2d_debug_detailed = False

# 2D runtime state
ema_vpi = None
ema_lfc = None
ema_nmi = None
pose2d_state = "normal"
pose2d_last_state_change_frame_index = -10**9
pose2d_none_run = 0
pose2d_ear_consec_run = 0
pose2d_recent_flags = deque(maxlen=5)


def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def _eye_aspect_ratio(lm, ids):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in ids]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * (h + 1e-6)))


def _avg_ear(lm):
    try:
        left_ids = [36, 37, 38, 39, 40, 41]
        right_ids = [42, 43, 44, 45, 46, 47]
        ear_l = _eye_aspect_ratio(lm, left_ids)
        ear_r = _eye_aspect_ratio(lm, right_ids)
        return 0.5 * (ear_l + ear_r)
    except Exception:
        return 1.0  # non-extreme default


def compute_2d_metrics(lm, face_bbox):
    """Compute 2D ratios that correlate with down-pitch.
    lm: (68,2) landmark array; face_bbox: (x1,y1,x2,y2)
    Returns: vpi, lfc, nmi, avg_ear
    """
    # key points
    brow_mid = 0.5 * (lm[21] + lm[22])
    nose = lm[30]
    chin = lm[8]
    mouth = 0.5 * (lm[62] + lm[66])

    face_h = max(1.0, float(face_bbox[3] - face_bbox[1]))
    denom_bn = max(1.0, float(nose[1] - brow_mid[1]))

    vpi = float((chin[1] - nose[1]) / denom_bn)
    lfc = float((chin[1] - mouth[1]) / face_h)
    nmi = float((mouth[1] - nose[1]) / denom_bn)
    ear = float(_avg_ear(lm))
    return vpi, lfc, nmi, ear


def should_filter_frame_by_2d(vpi, lfc, nmi, ear, frame_index):
    """AND logic on smoothed ratios + consecutive frames + min-hold + conservative none fallback.
    Returns True if we should filter (use original frame).
    """
    global ema_vpi, ema_lfc, ema_nmi, pose2d_recent_flags
    global pose2d_state, pose2d_last_state_change_frame_index
    global pose2d_none_run, pose2d_ear_consec_run
    global pose2d_vpi_thr, pose2d_lfc_thr, pose2d_nmi_thr
    global pose2d_ema_alpha, pose2d_consec_frames, pose2d_min_hold_frames
    global pose2d_none_consecutive_max, pose2d_enable_ear_gate
    global pose2d_ear_thr, pose2d_ear_gate_consec, pose2d_ear_bias
    global pose2d_debug_detailed

    # Update EAR corroboration counter
    if pose2d_enable_ear_gate and ear < pose2d_ear_thr:
        pose2d_ear_consec_run += 1
    else:
        pose2d_ear_consec_run = 0

    # Conservative fallback: if metrics are invalid, treat as none
    metrics_valid = np.isfinite(vpi) and np.isfinite(lfc) and np.isfinite(nmi)
    if not metrics_valid:
        pose2d_none_run += 1
        if pose2d_debug_detailed:
            print(f"[Pose2D] f={frame_index} none_run={pose2d_none_run} state={pose2d_state} dec={'FILTER' if pose2d_state=='filtered' else 'KEEP'} reason=invalid_metrics")
        # If none frames exceed threshold, force FILTER
        if pose2d_none_run >= pose2d_none_consecutive_max:
            pose2d_state = 'filtered'
        return pose2d_state == 'filtered'
    else:
        pose2d_none_run = 0

    # EMA smoothing
    ema_vpi = vpi if ema_vpi is None else (pose2d_ema_alpha * vpi + (1 - pose2d_ema_alpha) * ema_vpi)
    ema_lfc = lfc if ema_lfc is None else (pose2d_ema_alpha * lfc + (1 - pose2d_ema_alpha) * ema_lfc)
    ema_nmi = nmi if ema_nmi is None else (pose2d_ema_alpha * nmi + (1 - pose2d_ema_alpha) * ema_nmi)

    # Apply optional EAR bias (makes entry a bit easier when squinting sustained)
    vpi_thr = pose2d_vpi_thr
    lfc_thr = pose2d_lfc_thr
    if pose2d_enable_ear_gate and pose2d_ear_consec_run >= pose2d_ear_gate_consec:
        vpi_thr += pose2d_ear_bias
        lfc_thr += pose2d_ear_bias

    # AND logic: both VPI and LFC must indicate extreme (NMI logged only)
    extreme_and = (ema_vpi <= vpi_thr) and (ema_lfc <= lfc_thr)

    # Enforce consecutive agreement and min-hold
    pose2d_recent_flags.append(extreme_and)
    if len(pose2d_recent_flags) < pose2d_consec_frames:
        if pose2d_debug_detailed:
            print(f"[Pose2D] f={frame_index} VPI={ema_vpi:.2f} LFC={ema_lfc:.3f} NMI={ema_nmi:.2f} EAR={ear:.2f} OR=AND={extreme_and} state={pose2d_state} dec={'FILTER' if pose2d_state=='filtered' else 'KEEP'} reason=warmup")
        return pose2d_state == 'filtered'

    all_recent_extreme = all(list(pose2d_recent_flags)[-pose2d_consec_frames:])
    frames_since_change = frame_index - pose2d_last_state_change_frame_index
    can_change = frames_since_change >= pose2d_min_hold_frames

    before = pose2d_state
    after = pose2d_state

    if can_change:
        if pose2d_state == 'normal' and all_recent_extreme:
            after = 'filtered'
            pose2d_last_state_change_frame_index = frame_index
        elif pose2d_state == 'filtered' and (not extreme_and) and (not any(list(pose2d_recent_flags)[-pose2d_consec_frames:])):
            # recent window all non-extreme -> exit
            after = 'normal'
            pose2d_last_state_change_frame_index = frame_index

    pose2d_state = after
    decision_filter = (after == 'filtered')

    if pose2d_debug_detailed:
        reasons = []
        if not can_change:
            reasons.append(f"hold({pose2d_min_hold_frames - frames_since_change} left)" if frames_since_change < pose2d_min_hold_frames else "hold(0)")
        if all_recent_extreme:
            reasons.append("extreme")
        else:
            reasons.append("nonextreme")
        if before != after:
            reasons.append("state_changed")
        print(
            f"[Pose2D] f={frame_index} VPI={ema_vpi:.2f} LFC={ema_lfc:.3f} NMI={ema_nmi:.2f} EAR={ear:.2f} "
            f"AND={'T' if extreme_and else 'F'} state={before}->{after} dec={'FILTER' if decision_filter else 'KEEP'} reason={'+'.join(reasons)}"
        )

    return decision_filter

def set_pose2d_filter_config(enabled=False,
                              vpi_thr=1.20,
                              lfc_thr=0.20,
                              nmi_thr=0.55,
                              ema_alpha=0.30,
                              consec_frames=3,
                              min_hold_frames=6,
                              none_consecutive_max=4,
                              enable_ear_gate=False,
                              ear_thr=0.18,
                              ear_gate_consec=2,
                              ear_bias=0.02,
                              debug_detailed=False):
    """Configure 2D-only pose filtering (no angles).

    AND logic: VPI and LFC must indicate extreme to enter FILTER (NMI logged only).
    """
    global pose2d_filter_enabled, pose2d_vpi_thr, pose2d_lfc_thr, pose2d_nmi_thr
    global pose2d_ema_alpha, pose2d_consec_frames, pose2d_min_hold_frames
    global pose2d_none_consecutive_max, pose2d_enable_ear_gate
    global pose2d_ear_thr, pose2d_ear_gate_consec, pose2d_ear_bias
    global pose2d_debug_detailed

    pose2d_filter_enabled = bool(enabled)
    pose2d_vpi_thr = float(vpi_thr)
    pose2d_lfc_thr = float(lfc_thr)
    pose2d_nmi_thr = float(nmi_thr)
    pose2d_ema_alpha = float(ema_alpha)
    pose2d_consec_frames = int(consec_frames)
    pose2d_min_hold_frames = int(min_hold_frames)
    pose2d_none_consecutive_max = int(none_consecutive_max)
    pose2d_enable_ear_gate = bool(enable_ear_gate)
    pose2d_ear_thr = float(ear_thr)
    pose2d_ear_gate_consec = int(ear_gate_consec)
    pose2d_ear_bias = float(ear_bias)
    pose2d_debug_detailed = bool(debug_detailed)

    if pose2d_filter_enabled:
        print(
            f"Pose2D filtering enabled (AND): VPI<={pose2d_vpi_thr:.2f} & LFC<={pose2d_lfc_thr:.2f} (NMI logged only), "
            f"ema={pose2d_ema_alpha}, consec={pose2d_consec_frames}, hold={pose2d_min_hold_frames}, "
            f"none->filter after {pose2d_none_consecutive_max} frames, "
            f"EARgate={'on' if pose2d_enable_ear_gate else 'off'} (thr={pose2d_ear_thr}, k={pose2d_ear_gate_consec}, bias={pose2d_ear_bias}), "
            f"debug={'on' if pose2d_debug_detailed else 'off'}"
        )
        if pitch_filter_enabled:
            print("Note: Both Pose2D and angle-based filters are enabled; frame will be filtered if either detects extreme pose.")

def set_pitch_filter_config(enabled=False, down_threshold=60.0, up_threshold=55.0,
                            conf_min=0.6, ema_alpha=0.3, min_hold_frames=6, debug_detailed=False):
    """Configure pitch filtering parameters (Option 1)."""
    global pitch_filter_enabled, pitch_down_threshold_deg, pitch_up_threshold_deg
    global pitch_conf_min, pitch_ema_alpha, pitch_min_hold_frames, pitch_debug_detailed
    pitch_filter_enabled = enabled
    pitch_down_threshold_deg = down_threshold
    pitch_up_threshold_deg = up_threshold
    pitch_conf_min = conf_min
    pitch_ema_alpha = ema_alpha
    pitch_min_hold_frames = int(min_hold_frames)
    pitch_debug_detailed = bool(debug_detailed)
    if enabled:
        print(
            f"Pitch filtering enabled: down_threshold={down_threshold}°, up_threshold={up_threshold}°, "
            f"conf_min={pitch_conf_min}, ema_alpha={pitch_ema_alpha}, min_hold_frames={min_hold_frames}, "
            f"debug={'on' if pitch_debug_detailed else 'off'}"
        )

def reset_pitch_filter_state():
    """Reset filtering state (angles and 2D) for new video processing"""
    global pitch_history, pitch_state, ema_pitch, last_state_change_frame_index
    global ema_vpi, ema_lfc, ema_nmi, pose2d_state, pose2d_recent_flags
    global pose2d_last_state_change_frame_index, pose2d_none_run, pose2d_ear_consec_run

    # Angle-based
    pitch_history.clear()
    pitch_state = "normal"
    ema_pitch = None
    last_state_change_frame_index = -10**9

    # 2D-based
    ema_vpi = None
    ema_lfc = None
    ema_nmi = None
    pose2d_state = "normal"
    pose2d_recent_flags.clear()
    pose2d_last_state_change_frame_index = -10**9
    pose2d_none_run = 0
    pose2d_ear_consec_run = 0

def compute_pitch_from_landmarks(landmarks, img_width, img_height):
    """
    Compute head pitch angle from facial landmarks using PnP (RANSAC) and derive a
    confidence from reprojection error.

    Args:
        landmarks: 68 facial landmarks in iBUG format (numpy array of shape [68, 2])
        img_width: Image width
        img_height: Image height

    Returns:
        (pitch_degrees, confidence):
            - pitch_degrees is head pitch angle in degrees (positive = looking down)
            - confidence in [0,1], higher means more reliable pose
    """
    try:
        # Validate landmarks
        if landmarks.shape[0] != 68 or landmarks.shape[1] != 2:
            return None, 0.0

        # Basic landmark quality checks
        landmark_std = np.std(landmarks, axis=0)
        if landmark_std[0] < 5 or landmark_std[1] < 5:  # Too clustered
            return None, 0.0

        # Key landmarks must be in-bounds
        key_indices = [30, 8, 36, 45, 48, 54]  # nose tip, chin, eye corners, mouth corners
        for idx in key_indices:
            if idx >= len(landmarks):
                return None, 0.0
            x, y = landmarks[idx]
            if x < 0 or y < 0 or x >= img_width or y >= img_height:
                return None, 0.0

        # 2D-3D correspondences
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye outer corner
            landmarks[45],  # Right eye outer corner
            landmarks[48],  # Left mouth corner
            landmarks[54],  # Right mouth corner
        ], dtype=np.float32)

        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [0.0, -90.0, -10.0],    # Chin
            [-60.0, 40.0, -30.0],   # Left eye outer corner
            [60.0, 40.0, -30.0],    # Right eye outer corner
            [-40.0, -30.0, -30.0],  # Left mouth corner
            [40.0, -30.0, -30.0],   # Right mouth corner
        ], dtype=np.float32)

        # Camera intrinsics (approximate)
        focal_length = float(img_width)
        center = (float(img_width) / 2.0, float(img_height) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Robust PnP with RANSAC for better stability at extremes
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            model_points, image_points, camera_matrix, None,
            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None, 0.0

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Pitch angle (rotation around x-axis)
        pitch_radians = np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)
        )
        pitch_degrees = float(np.degrees(pitch_radians))

        # Compute reprojection error and derive confidence in [0,1]
        projected_points, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, None)
        projected_points = projected_points.reshape(-1, 2)
        diffs = projected_points - image_points
        dists = np.linalg.norm(diffs, axis=1)
        mean_err_px = float(np.mean(dists))

        # Normalize error by inter-ocular distance for scale invariance
        inter_ocular = float(np.linalg.norm(landmarks[45] - landmarks[36]))
        norm_err = mean_err_px / (inter_ocular + 1e-6)
        confidence = 1.0 / (1.0 + norm_err)  # higher is better
        confidence = float(max(0.0, min(1.0, confidence)))

        return pitch_degrees, confidence

    except Exception as e:
        print(f"Error computing pitch: {e}")
        return None, 0.0

def should_filter_frame_by_pitch(pitch_degrees, confidence, frame_index):
    """
    Determine if frame should be filtered based on smoothed pitch with hysteresis,
    confidence gating, and a minimum hold time to reduce flicker.

    Args:
        pitch_degrees: Current frame raw pitch angle (deg)
        confidence: Confidence in [0,1] from reprojection error
        frame_index: Monotonic frame counter (int)

    Returns:
        bool: True if frame should be filtered (use original frame)
    """
    global pitch_history, pitch_state, ema_pitch, pitch_ema_alpha
    global last_state_change_frame_index, pitch_min_hold_frames, pitch_conf_min
    global pitch_down_threshold_deg, pitch_up_threshold_deg, consecutive_frames_threshold
    global pitch_debug_detailed

    # If pitch cannot be computed, hold current state
    if pitch_degrees is None:
        if pitch_debug_detailed:
            print(f"[PitchDebug] frame={frame_index} pitch=None conf={confidence:.2f} state={pitch_state} decision={'FILTER' if pitch_state=='filtered' else 'KEEP'} reason=no_measurement")
        return pitch_state == "filtered"

    # Update EMA smoothing
    if ema_pitch is None:
        ema_pitch = float(pitch_degrees)
    else:
        ema_pitch = float(pitch_ema_alpha * pitch_degrees + (1.0 - pitch_ema_alpha) * ema_pitch)

    # Maintain history of smoothed pitch for short consecutive-window stability
    pitch_history.append(ema_pitch)

    # Default decision: hold current state until enough history
    if len(pitch_history) < consecutive_frames_threshold:
        if pitch_debug_detailed:
            print(f"[PitchDebug] frame={frame_index} raw={pitch_degrees:.1f} ema={ema_pitch:.1f} conf={confidence:.2f} state={pitch_state} decision={'FILTER' if pitch_state=='filtered' else 'KEEP'} reason=warmup")
        return pitch_state == "filtered"

    recent_pitches = list(pitch_history)[-consecutive_frames_threshold:]

    # Enforce minimum hold time after a state change
    frames_since_change = frame_index - last_state_change_frame_index
    can_change = frames_since_change >= pitch_min_hold_frames

    state_before = pitch_state
    state_after = pitch_state

    # Confidence gating: only allow transitions when we trust the measurement
    if confidence >= pitch_conf_min and can_change:
        if pitch_state == "normal":
            if all(p >= pitch_down_threshold_deg for p in recent_pitches):
                state_after = "filtered"
                last_state_change_frame_index = frame_index
        else:  # filtered -> normal
            if all(p <= pitch_up_threshold_deg for p in recent_pitches):
                state_after = "normal"
                last_state_change_frame_index = frame_index

    # Build decision
    decision_filter = (state_after == "filtered")
    pitch_state = state_after

    if pitch_debug_detailed:
        reason = []
        if not can_change:
            reason.append(f"hold({pitch_min_hold_frames - frames_since_change} left)" if frames_since_change < pitch_min_hold_frames else "hold(0)")
        if confidence < pitch_conf_min:
            reason.append("low_conf")
        if state_before != state_after:
            reason.append("state_changed")
        if not reason:
            reason.append("stable")
        print(
            f"[PitchDebug] frame={frame_index} raw={pitch_degrees:.1f} ema={ema_pitch:.1f} conf={confidence:.2f} "
            f"thr_down={pitch_down_threshold_deg:.1f} thr_up={pitch_up_threshold_deg:.1f} "
            f"state={state_before}->{state_after} decision={'FILTER' if decision_filter else 'KEEP'} reason={'+'.join(reason)}"
        )

    return decision_filter

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range



def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')

    filters_enabled = []
    if pose2d_filter_enabled:
        filters_enabled.append('Pose2D filtering (AND)')
    if pitch_filter_enabled:
        filters_enabled.append('Pitch filtering (angles)')
    if filters_enabled:
        print(f'{" + ".join(filters_enabled)} enabled - extreme downward poses will use original frames')

    average_range_minus = []
    average_range_plus = []
    pitch_filtered_count = 0

    frame_index = 0
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            frame_index += 1


            # Apply filtering - both filters can run simultaneously
            filter_by_2d = False
            filter_by_angle = False

            if pose2d_filter_enabled:
                vpi, lfc, nmi, ear = compute_2d_metrics(face_land_mark, f)
                filter_by_2d = should_filter_frame_by_2d(vpi, lfc, nmi, ear, frame_index)

            if pitch_filter_enabled:
                frame_height, frame_width = fb[j].shape[:2]
                pitch_degrees, pitch_confidence = compute_pitch_from_landmarks(face_land_mark, frame_width, frame_height)
                filter_by_angle = should_filter_frame_by_pitch(pitch_degrees, pitch_confidence, frame_index)

            # Combine decisions (OR logic - filter if either says to filter)
            if filter_by_2d or filter_by_angle:
                coords_list += [coord_placeholder]
                pitch_filtered_count += 1
                continue

            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark

            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]

    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0}~{int(sum(average_range_plus) / len(average_range_plus)) if average_range_plus else 0} ] , the current value: {upperbondrange}")
    if pitch_filtered_count > 0:
        filter_reasons = []
        if pose2d_filter_enabled:
            filter_reasons.append(f"Pose2D (AND logic: VPI<={pose2d_vpi_thr}, LFC<={pose2d_lfc_thr})")
        if pitch_filter_enabled:
            filter_reasons.append(f"Pitch angles (≥{pitch_down_threshold_deg}°)")
        if filter_reasons:
            print(f"Combined filtering: {pitch_filtered_count} frames filtered due to extreme downward pose ({' OR '.join(filter_reasons)})")
    print("*************************************************************************************************************************************")
    return coords_list, frames, pitch_filtered_count


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames, pitch_filtered_count = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)

        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)

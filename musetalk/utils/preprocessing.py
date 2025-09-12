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
pitch_history = deque(maxlen=5)  # Store last 5 pitch values for smoothing
pitch_state = "normal"  # "normal" or "filtered"
consecutive_frames_threshold = 2

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def set_pitch_filter_config(enabled=False, down_threshold=60.0, up_threshold=55.0):
    """Configure pitch filtering parameters"""
    global pitch_filter_enabled, pitch_down_threshold_deg, pitch_up_threshold_deg
    pitch_filter_enabled = enabled
    pitch_down_threshold_deg = down_threshold
    pitch_up_threshold_deg = up_threshold
    if enabled:
        print(f"Pitch filtering enabled: down_threshold={down_threshold}°, up_threshold={up_threshold}°")

def reset_pitch_filter_state():
    """Reset pitch filtering state for new video processing"""
    global pitch_history, pitch_state
    pitch_history.clear()
    pitch_state = "normal"

def compute_pitch_from_landmarks(landmarks, img_width, img_height):
    """
    Compute head pitch angle from facial landmarks using PnP algorithm.

    Args:
        landmarks: 68 facial landmarks in iBUG format (numpy array of shape [68, 2])
        img_width: Image width
        img_height: Image height

    Returns:
        pitch_degrees: Head pitch angle in degrees (positive = looking down)
    """
    try:
        # Check if landmarks are valid
        if landmarks.shape[0] != 68 or landmarks.shape[1] != 2:
            return None

        # Check for landmark quality - detect if landmarks are too clustered or extreme
        landmark_std = np.std(landmarks, axis=0)
        if landmark_std[0] < 5 or landmark_std[1] < 5:  # Too clustered
            return None

        # Check if key landmarks are within reasonable bounds
        key_indices = [30, 8, 36, 45, 48, 54]  # nose tip, chin, eye corners, mouth corners
        for idx in key_indices:
            if idx >= len(landmarks):
                return None
            x, y = landmarks[idx]
            if x < 0 or y < 0 or x >= img_width or y >= img_height:
                return None

        # Map iBUG-68 landmarks to 3D model points
        # Using 6 key points: nose tip, chin, eye corners, mouth corners
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye outer corner
            landmarks[45],  # Right eye outer corner
            landmarks[48],  # Left mouth corner
            landmarks[54],  # Right mouth corner
        ], dtype=np.float32)

        # 3D model points in mm (standard canonical face model)
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Nose tip
            [0.0, -90.0, -10.0],    # Chin
            [-60.0, 40.0, -30.0],   # Left eye outer corner
            [60.0, 40.0, -30.0],    # Right eye outer corner
            [-40.0, -30.0, -30.0],  # Left mouth corner
            [40.0, -30.0, -30.0],   # Right mouth corner
        ], dtype=np.float32)

        # Camera intrinsics (approximate)
        focal_length = img_width
        center = (img_width / 2, img_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract pitch angle (rotation around x-axis)
        # pitch = arctan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
        pitch_radians = np.arctan2(-rotation_matrix[2, 0],
                                 np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        pitch_degrees = np.degrees(pitch_radians)

        return pitch_degrees

    except Exception as e:
        print(f"Error computing pitch: {e}")
        return None

def should_filter_frame_by_pitch(pitch_degrees):
    """
    Determine if frame should be filtered based on pitch angle with hysteresis.

    Args:
        pitch_degrees: Current frame pitch angle

    Returns:
        bool: True if frame should be filtered (use original frame)
    """
    global pitch_history, pitch_state

    if pitch_degrees is None:
        return False  # Don't filter if we can't compute pitch

    # Add to history
    pitch_history.append(pitch_degrees)

    # Need at least a few frames for stable decision
    if len(pitch_history) < consecutive_frames_threshold:
        return pitch_state == "filtered"

    # Get recent pitch values
    recent_pitches = list(pitch_history)[-consecutive_frames_threshold:]

    # State machine with hysteresis
    if pitch_state == "normal":
        # Check if we should enter filtered state
        if all(p >= pitch_down_threshold_deg for p in recent_pitches):
            pitch_state = "filtered"
            return True
        return False
    else:  # pitch_state == "filtered"
        # Check if we should exit filtered state
        if all(p <= pitch_up_threshold_deg for p in recent_pitches):
            pitch_state = "normal"
            return False
        return True

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

    if pitch_filter_enabled:
        print('Pitch filtering enabled - extreme downward poses will use original frames')

    average_range_minus = []
    average_range_plus = []
    pitch_filtered_count = 0

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

            # Check pitch angle if filtering is enabled
            if pitch_filter_enabled:
                frame_height, frame_width = fb[j].shape[:2]
                pitch_degrees = compute_pitch_from_landmarks(face_land_mark, frame_width, frame_height)

                if should_filter_frame_by_pitch(pitch_degrees):
                    coords_list += [coord_placeholder]
                    pitch_filtered_count += 1
                    # Optional debug logging (uncomment for troubleshooting)
                    # print(f"Frame filtered: pitch={pitch_degrees:.1f}° (threshold={pitch_down_threshold_deg}°)")
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
    if pitch_filter_enabled and pitch_filtered_count > 0:
        print(f"Pitch filtering: {pitch_filtered_count} frames filtered due to extreme downward pose (≥{pitch_down_threshold_deg}°)")
    print("*************************************************************************************************************************************")
    return coords_list,frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
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

import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys

from PIL import Image

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, set_pitch_filter_config, set_pose2d_filter_config, reset_pitch_filter_state

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)

    # Configure pitch filtering for normal inference mode
    set_pitch_filter_config(
        enabled=args.enable_pitch_filter,
        down_threshold=args.pitch_down_threshold,
        up_threshold=args.pitch_up_threshold,
        conf_min=args.pitch_conf_min,
        ema_alpha=args.pitch_ema_alpha,
        min_hold_frames=args.pitch_min_hold_frames,
        debug_detailed=args.pitch_debug_detailed
    )

    # Configure 2D pose-only filtering (overrides angle-based when enabled)
    set_pose2d_filter_config(
        enabled=getattr(args, 'enable_pose2d_filter', False),
        vpi_thr=getattr(args, 'pose2d_vpi_thr', 1.10),
        lfc_thr=getattr(args, 'pose2d_lfc_thr', 0.19),
        nmi_thr=getattr(args, 'pose2d_nmi_thr', 0.55),
        ema_alpha=getattr(args, 'pose2d_ema_alpha', 0.30),
        consec_frames=getattr(args, 'pose2d_consec_frames', 3),
        min_hold_frames=getattr(args, 'pose2d_min_hold_frames', 6),
        none_consecutive_max=getattr(args, 'pose2d_none_consecutive_max', 4),
        enable_ear_gate=getattr(args, 'pose2d_enable_ear_gate', False),
        ear_thr=getattr(args, 'pose2d_ear_thr', 0.18),
        ear_gate_consec=getattr(args, 'pose2d_ear_gate_consec', 2),
        ear_bias=getattr(args, 'pose2d_ear_bias', 0.02),
        debug_detailed=getattr(args, 'pose2d_debug_detailed', False),
    )

    # Process each task
    for task_id in inference_config:
        try:
            # Reset pitch filtering state for each new video
            reset_pitch_filter_state()

            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            if "result_name" in inference_config[task_id]:
                args.output_vid_name = inference_config[task_id]["result_name"]

            # Set bbox_shift based on version
            if args.version == "v15":
                bbox_shift = 0  # v15 uses fixed bbox_shift
            else:
                bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)  # v1 uses config or default

            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"

            # Create temporary directories
            temp_dir = args.result_dir
            os.makedirs(temp_dir, exist_ok=True)

            # Set result save paths
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)

            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            output_vid_name_concat = os.path.join(temp_dir, output_basename + "_concat.mp4")

            # Extract frames from source video
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

            # Extract audio features
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )

            # Preprocess input images
            pitch_filtered_count = 0  # Initialize pitch filtering counter
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, frame_list, pitch_filtered_count = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)

            print(f"Number of frames: {len(frame_list)}")

            # Process each frame
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                # Skip frames with invalid bbox dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)

            # Smooth first and last frames
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Get video length from audio chunks
            video_num = len(whisper_chunks)

            # Handle case where no valid faces were detected in any frame
            if len(input_latent_list) == 0:
                print("No valid face detections found. Using passthrough mode - writing original frames.")
                for i in range(video_num):
                    ori_frame = frame_list_cycle[i % len(frame_list_cycle)]
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
            else:
                # Initialize counter for no-lip bypass tracking
                no_lip_bypass_count = 0
                # Batch inference
                print("Starting inference")
                batch_size = args.batch_size
                gen = datagen(
                    whisper_chunks=whisper_chunks,
                    vae_encode_latents=input_latent_list_cycle,
                    batch_size=batch_size,
                    delay_frame=0,
                    device=device,
                )

                res_frame_list = []
                total = int(np.ceil(float(video_num) / batch_size))

                # Execute inference
                for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=unet.model.dtype)

                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)

                # Pad generated images to original video size
                print("Padding generated images to original video size")
                res_frame_idx = 0  # Track index in res_frame_list separately

                for i in range(video_num):
                    bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                    ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                    x1, y1, x2, y2 = bbox

                    # Handle frames with no face detection (placeholder or invalid bbox)
                    if bbox == coord_placeholder or x2 <= x1 or y2 <= y1:
                        # Write original frame unchanged
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                        continue

                    # Get the corresponding generated frame
                    if res_frame_idx >= len(res_frame_list):
                        # If we run out of generated frames, write original
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                        continue

                    res_frame = res_frame_list[res_frame_idx]
                    res_frame_idx += 1

                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, ori_frame.shape[0])  # Fixed: use ori_frame instead of frame

                    # Simple 'no lip' presence check on the face crop; bypass if lips are not detected
                    if getattr(args, 'enable_no_lip_bypass', False):
                        try:
                            crop_for_parse = ori_frame[y1:y2, x1:x2]
                            lip_mask_img = fp(Image.fromarray(crop_for_parse), mode="lips_only")
                            lip_mask_np = np.array(lip_mask_img)
                            lip_pixels = int(np.count_nonzero(lip_mask_np))
                            lip_frac = lip_pixels / float(lip_mask_np.size) if lip_mask_np.size > 0 else 0.0
                            if lip_pixels == 0 or lip_frac < getattr(args, 'no_lip_min_frac', 0.0015):
                                # Use original frame unchanged when no lips are detected
                                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                                no_lip_bypass_count += 1
                                continue
                        except Exception as e:
                            # Be permissive on error; proceed with normal pipeline
                            pass
                    try:
                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    except:
                        # If resize fails, write original frame
                        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", ori_frame)
                        continue

                    # Merge results with version-specific parameters
                    if args.version == "v15":
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                    else:
                        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                    cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

                # Report pitch filtering results
                if args.enable_pitch_filter:
                    print(f"Pitch filtering: {pitch_filtered_count} frames skipped due to pitch detection criteria")

                # Report no-lip bypass results
                if getattr(args, 'enable_no_lip_bypass', False) and no_lip_bypass_count > 0:
                    print(f"No-lip bypass: {no_lip_bypass_count} frames used original frames due to insufficient lip detection")

            # Save prediction results
            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            print("Video generation command:", cmd_img2video)
            os.system(cmd_img2video)

            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
            print("Audio combination command:", cmd_combine_audio)
            os.system(cmd_combine_audio)

            # Clean up temporary files
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)

            shutil.rmtree(save_dir_full)
            if not args.saved_coord:
                os.remove(crop_coord_save_path)

            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--enable_no_lip_bypass", action="store_true", help="Bypass inference when lips are not detected in face crop")
    parser.add_argument("--no_lip_min_frac", type=float, default=0.0015, help="Minimum fraction of lip pixels (512x512 parsing) to accept frame")

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    # Pose2D-only filtering (2D ratios + simple hysteresis)
    parser.add_argument("--enable_pose2d_filter", action="store_true", help="Enable 2D landmarks-only pose filtering (overrides angle-based when enabled)")
    parser.add_argument("--pose2d_vpi_thr", type=float, default=1.10, help="Threshold for VPI: (chin_y - nose_y) / (nose_y - brow_mid_y). Filter when VPI <= threshold (lower indicates down)")
    parser.add_argument("--pose2d_lfc_thr", type=float, default=0.19, help="Threshold for LFC: (chin_y - mouth_y) / face_bbox_h. Filter when LFC <= threshold (lower indicates down)")
    parser.add_argument("--pose2d_nmi_thr", type=float, default=0.55, help="Threshold for NMI: (mouth_y - nose_y) / (nose_y - brow_mid_y). Not used in decision; logged only")
    parser.add_argument("--pose2d_ema_alpha", type=float, default=0.30, help="EMA alpha for ratio smoothing (0-1)")
    parser.add_argument("--pose2d_consec_frames", type=int, default=3, help="Consecutive frames required to enter/exit state")
    parser.add_argument("--pose2d_min_hold_frames", type=int, default=6, help="Minimum frames to hold state before allowing change")
    parser.add_argument("--pose2d_none_consecutive_max", type=int, default=4, help="None/invalid metrics for N frames forces FILTER (conservative)")
    parser.add_argument("--pose2d_enable_ear_gate", action="store_true", help="Enable optional EAR gate to boost down decision when squinting sustained")
    parser.add_argument("--pose2d_ear_thr", type=float, default=0.18, help="EAR threshold for squint detection")
    parser.add_argument("--pose2d_ear_gate_consec", type=int, default=2, help="Consecutive frames of low EAR to trigger bias")
    parser.add_argument("--pose2d_ear_bias", type=float, default=0.02, help="Bias added to VPI and NMI thresholds when EAR gate is on")
    parser.add_argument("--pose2d_debug_detailed", action="store_true", help="Enable per-frame detailed Pose2D filter debug logs")

    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--pitch_conf_min", type=float, default=0.6, help="Minimum confidence (0-1) required to change pitch filter state")
    parser.add_argument("--pitch_ema_alpha", type=float, default=0.3, help="EMA alpha for pitch smoothing (0-1)")
    parser.add_argument("--pitch_min_hold_frames", type=int, default=6, help="Minimum frames to hold state before allowing change")
    parser.add_argument("--pitch_debug_detailed", action="store_true", help="Enable per-frame detailed pitch filter debug logs")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    parser.add_argument("--enable_pitch_filter", action="store_true", help="Enable pitch filtering for extreme downward poses")
    parser.add_argument("--pitch_down_threshold", type=float, default=30.0, help="Pitch angle threshold for filtering (degrees downward)")
    parser.add_argument("--pitch_up_threshold", type=float, default=20.0, help="Pitch angle threshold for exiting filter (degrees downward)")
    args = parser.parse_args()
    main(args)

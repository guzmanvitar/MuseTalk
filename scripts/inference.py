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
import time
from contextlib import contextmanager
import tempfile
import traceback

# Optional memory utilities
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import resource  # Unix-only
except Exception:
    resource = None


def _cpu_mem_mb():
    try:
        if psutil is not None:
            proc = psutil.Process(os.getpid())
            return proc.memory_info().rss / (1024 * 1024)
        if resource is not None:
            # ru_maxrss is KB on Linux, bytes on macOS. Treat as KB here (Linux most likely)
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(rss_kb) / 1024.0
    except Exception:
        pass
    return -1.0


def _gpu_mem_mb(device):
    try:
        if torch.cuda.is_available():
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            torch.cuda.synchronize(dev)
            allocated = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            return allocated, reserved
    except Exception:
        pass
    return -1.0, -1.0


class Perf:
    """Lightweight per-section wall-time and memory profiler."""
    def __init__(self, device):
        self.device = device
        self.sections = {}
        self._starts = {}
        self.t0 = time.perf_counter()
        print(f"[PROF] Profiling enabled. CUDA={torch.cuda.is_available()} device={device}")

    def start(self, name):
        self._starts[name] = time.perf_counter()
        cpu_mb = _cpu_mem_mb()
        gpu_alloc, gpu_res = _gpu_mem_mb(self.device)
        print(f"[PROF][START] {name} | t={self._starts[name]-self.t0:8.3f}s | CPU={cpu_mb:.1f}MB | GPU alloc/res={gpu_alloc:.1f}/{gpu_res:.1f}MB")

    def end(self, name, extra_note: str = ""):
        t1 = time.perf_counter()
        t0 = self._starts.get(name, t1)
        dt = t1 - t0
        self.sections[name] = self.sections.get(name, 0.0) + dt
        cpu_mb = _cpu_mem_mb()
        gpu_alloc, gpu_res = _gpu_mem_mb(self.device)
        note = f" | {extra_note}" if extra_note else ""
        print(f"[PROF][END]   {name} | dt={dt:8.3f}s | CPU={cpu_mb:.1f}MB | GPU alloc/res={gpu_alloc:.1f}/{gpu_res:.1f}MB{note}")

    def report(self, total_media_seconds: float | None = None):
        total = sum(self.sections.values())
        print("\n===== Performance Breakdown =====")
        for k, v in sorted(self.sections.items(), key=lambda x: -x[1]):
            pct = (100.0 * v / total) if total > 0 else 0.0
            print(f"{k:35s}: {v:8.3f}s  ({pct:5.1f}%)")
        print(f"Total measured sections            : {total:8.3f}s")
        if total_media_seconds is not None and total_media_seconds > 0:
            rtf = total / total_media_seconds
            print(f"Input media duration               : {total_media_seconds:8.3f}s")
            print(f"Real-time factor (RTF)             : {rtf:8.3f}x slower than real-time")

        print("==================================\n")

# ========================= Parallelization Coordinator Helpers =========================

def detect_system(device):
    info = {
        "cpu_logical": os.cpu_count() or 1,
        "mem_total_gb": None,
        "mem_avail_gb": None,
        "gpu_name": None,
        "gpu_total_gb": None,
        "gpu_alloc_gb": None,
        "gpu_reserved_gb": None,
    }
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            info["mem_total_gb"] = vm.total / (1024 ** 3)
            info["mem_avail_gb"] = vm.available / (1024 ** 3)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            props = torch.cuda.get_device_properties(dev)
            info["gpu_name"] = props.name
            info["gpu_total_gb"] = props.total_memory / (1024 ** 3)
            alloc_mb, res_mb = _gpu_mem_mb(dev)
            info["gpu_alloc_gb"] = (alloc_mb / 1024.0) if alloc_mb >= 0 else None
            info["gpu_reserved_gb"] = (res_mb / 1024.0) if res_mb >= 0 else None
    except Exception:
        pass
    return info


def _estimate_video_duration_sec(video_path: str) -> float:
    try:
        if get_file_type(video_path) == "video":
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if frame_count > 0 and fps > 0:
                return float(frame_count) / float(fps)
        # directory of images or single image -> small default duration
        return 8.0
    except Exception:
        return 8.0


def _build_task_list(conf) -> list:
    tasks = []
    for task_id in conf:
        try:
            video_path = conf[task_id]["video_path"]
            audio_path = conf[task_id].get("audio_path")
            dur = _estimate_video_duration_sec(video_path)
            tasks.append({
                "task_id": task_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "dur": dur,
            })
        except Exception:
            continue
    return tasks


def _partition_tasks(tasks: list, n_bins: int):
    bins = [{"dur": 0.0, "items": []} for _ in range(max(1, n_bins))]
    for t in sorted(tasks, key=lambda x: -x.get("dur", 0.0)):
        b = min(bins, key=lambda x: x["dur"])
        b["items"].append(t)
        b["dur"] += t.get("dur", 0.0)
    return bins


def _write_subset_yaml(items: list, base_dir: str, idx: int) -> str:
    subset = {}
    for t in items:
        entry = {
            "video_path": t["video_path"],
        }
        if t.get("audio_path"):
            entry["audio_path"] = t["audio_path"]
        subset[t["task_id"]] = entry
    tmp_dir = os.path.join(base_dir, "_parallel_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    child_yaml = os.path.join(tmp_dir, f"subset_{idx}.yaml")
    OmegaConf.save(config=OmegaConf.create(subset), f=child_yaml)
    return child_yaml


def _parse_prof_from_log(log_path: str):
    tot_measured = 0.0
    media_secs = 0.0
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if "Total measured sections" in line:
                    try:
                        tot_measured = float(line.strip().split(":")[-1].strip().rstrip("s"))
                    except Exception:
                        pass
                if "Input media duration" in line:
                    try:
                        media_secs = float(line.strip().split(":")[-1].strip().rstrip("s"))
                    except Exception:
                        pass
    except Exception:
        pass
    return tot_measured, media_secs


def coordinator_main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    sys_info = detect_system(device)

    # Recommendations
    n = max(1, int(args.num_threads))
    cpu_logical = sys_info.get("cpu_logical", 1)
    t_cpu = args.cpu_threads_per_job if args.cpu_threads_per_job else max(2, min(8, int(0.8 * cpu_logical / n)))
    t_ff = max(1, t_cpu // 2)

    # Load config and partition tasks
    conf = OmegaConf.load(args.inference_config)
    tasks = _build_task_list(conf)
    bins = _partition_tasks(tasks, n)


    print("[COORD] System:", sys_info)
    print(f"[COORD] Spawning {n} jobs | cpu_threads/job={t_cpu} ffmpeg_threads/job={t_ff}")

    children = []
    logs = []
    metas = []
    for i, b in enumerate(bins):

        if len(b["items"]) == 0:
            continue
        child_yaml = _write_subset_yaml(b["items"], args.result_dir, i)
        child_result_dir = os.path.join(args.result_dir, f"p{i}")
        os.makedirs(child_result_dir, exist_ok=True)

        # Scale batch sizes conservatively for multi-job
        bs_child = max(4, int(args.batch_size if hasattr(args, "batch_size") else 8))
        vae_bs_child = max(8, int(getattr(args, "vae_batch_size", 32)))
        if n > 1:
            bs_child = max(6, int(bs_child * 0.8))
            vae_bs_child = max(16, int(vae_bs_child * 0.8))

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t_cpu)
        env["MKL_NUM_THREADS"] = str(t_cpu)
        env["OPENBLAS_NUM_THREADS"] = str(t_cpu)
        env["NUMEXPR_NUM_THREADS"] = str(t_cpu)

        log_path = os.path.join(child_result_dir, f"child_{i}.log")
        logs.append(log_path)
        metas.append({"i": i, "child_yaml": child_yaml, "child_result_dir": child_result_dir})

        logf = open(log_path, "w")

        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--inference_config", child_yaml,
            "--result_dir", child_result_dir,
            "--unet_model_path", args.unet_model_path,
            "--unet_config", args.unet_config,
            "--version", args.version,
            "--batch_size", str(bs_child),
            "--vae_batch_size", str(vae_bs_child),
            "--ffmpeg_preset", getattr(args, "ffmpeg_preset", "veryfast"),
        ]
        if getattr(args, "ffmpeg_threads", None):
            cmd += ["--ffmpeg_threads", str(args.ffmpeg_threads)]
        if getattr(args, "use_float16", False):
            cmd += ["--use_float16"]
        if getattr(args, "enable_no_lip_bypass", False):
            cmd += ["--enable_no_lip_bypass"]
        if getattr(args, "enable_pose2d_filter", False):
            cmd += ["--enable_pose2d_filter",
                    "--pose2d_vpi_thr", str(args.pose2d_vpi_thr),
                    "--pose2d_lfc_thr", str(args.pose2d_lfc_thr)]
        if getattr(args, "whisper_dir", None):
            cmd += ["--whisper_dir", args.whisper_dir]
        if getattr(args, "gpu_id", None) is not None:
            cmd += ["--gpu_id", str(args.gpu_id)]
        if getattr(args, "ffmpeg_path", None):
            cmd += ["--ffmpeg_path", args.ffmpeg_path]

        print("[COORD] Launch:", " ".join(cmd))
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf)
        children.append({"proc": p, "logf": logf})

        if i < len(bins) - 1 and args.stagger_start_secs > 0:
            time.sleep(args.stagger_start_secs)

    # Monitor and aggregate
    retcodes = []
    for child in children:
        rc = child["proc"].wait()
        try:
            child["logf"].flush(); child["logf"].close()
        except Exception:
            pass
        retcodes.append(rc)

    # Retry failed children once with reduced batch sizes (OOM resilience)
    if any(rc != 0 for rc in retcodes):
        print("[COORD] Some children failed. Attempting one retry with reduced batch sizes...")
        for idx, rc in enumerate(retcodes):
            if rc == 0:
                continue
            meta = metas[idx] if idx < len(metas) else None
            if not meta:
                continue
            i = meta["i"]
            child_yaml = meta["child_yaml"]
            child_result_dir = meta["child_result_dir"]
            retry_log_path = os.path.join(child_result_dir, f"child_{i}_retry.log")
            with open(retry_log_path, "w") as logf2:
                # recompute baseline child batch sizes
                bs_base = max(4, int(args.batch_size if hasattr(args, "batch_size") else 8))
                vae_base = max(8, int(getattr(args, "vae_batch_size", 32)))
                if n > 1:
                    bs_base = max(6, int(bs_base * 0.8))
                    vae_base = max(16, int(vae_base * 0.8))
                bs_retry = max(4, int(bs_base * getattr(args, "oom_retry_factor", 0.75)))
                vae_retry = max(8, int(vae_base * getattr(args, "oom_retry_factor", 0.75)))

                cmd2 = [
                    sys.executable, "-m", "scripts.inference",
                    "--inference_config", child_yaml,
                    "--result_dir", child_result_dir,
                    "--unet_model_path", args.unet_model_path,
                    "--unet_config", args.unet_config,
                    "--version", args.version,
                    "--batch_size", str(bs_retry),
                    "--vae_batch_size", str(vae_retry),
                    "--ffmpeg_preset", getattr(args, "ffmpeg_preset", "veryfast"),
                ]
                if getattr(args, "ffmpeg_threads", None):
                    cmd2 += ["--ffmpeg_threads", str(args.ffmpeg_threads)]
                if getattr(args, "use_float16", False):
                    cmd2 += ["--use_float16"]
                if getattr(args, "enable_no_lip_bypass", False):
                    cmd2 += ["--enable_no_lip_bypass"]
                if getattr(args, "enable_pose2d_filter", False):
                    cmd2 += ["--enable_pose2d_filter",
                            "--pose2d_vpi_thr", str(args.pose2d_vpi_thr),
                            "--pose2d_lfc_thr", str(args.pose2d_lfc_thr)]
                if getattr(args, "whisper_dir", None):
                    cmd2 += ["--whisper_dir", args.whisper_dir]
                if getattr(args, "gpu_id", None) is not None:
                    cmd2 += ["--gpu_id", str(args.gpu_id)]
                if getattr(args, "ffmpeg_path", None):
                    cmd2 += ["--ffmpeg_path", args.ffmpeg_path]

                print("[COORD][RETRY] Launch:", " ".join(cmd2))
                env2 = os.environ.copy()
                env2["OMP_NUM_THREADS"] = str(t_cpu)
                env2["MKL_NUM_THREADS"] = str(t_cpu)
                env2["OPENBLAS_NUM_THREADS"] = str(t_cpu)
                env2["NUMEXPR_NUM_THREADS"] = str(t_cpu)
                p2 = subprocess.Popen(cmd2, env=env2, stdout=logf2, stderr=logf2)
                rc2 = p2.wait()
            logs.append(retry_log_path)
            retcodes[idx] = rc2

    total_measured = 0.0
    total_media = 0.0
    for lp in logs:
        tm, ms = _parse_prof_from_log(lp)
        total_measured += tm
        total_media += ms

    print("[COORD] Children return codes:", retcodes)
    if total_media > 0:
        print(f"[COORD] Aggregated measured: {total_measured:.3f}s over media {total_media:.3f}s (RTF={total_measured/total_media:.2f}x)")
    else:
        print(f"[COORD] Aggregated measured: {total_measured:.3f}s")

    # Exit with non-zero if any child failed
    if any(rc != 0 for rc in retcodes):
        sys.exit(1)
    return




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

    # Coordinator mode: spawn multiple child processes and aggregate
    if getattr(args, "num_threads", 1) and int(args.num_threads) > 1:
        coordinator_main(args)
        return

    # Initialize profiler and global timers
    total_start = time.perf_counter()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    perf = Perf(device)
    acc_media_seconds = 0.0


    # Load model weights
    perf.start("init:load_models")
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    perf.end("init:load_models")

    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    perf.start("init:precision_and_to_device")

    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    # Infrastructure optimizations: channels_last for conv nets and cuDNN autotune
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        vae.vae = vae.vae.to(memory_format=torch.channels_last)
        unet.model = unet.model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    perf.end("init:precision_and_to_device")


    perf.start("init:audio_and_whisper")

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    perf.end("init:audio_and_whisper")


    # Initialize face parser with configurable parameters based on version
    perf.start("init:face_parsing")
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    perf.end("init:face_parsing")

    perf.start("init:config_and_filters")

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
    perf.end("init:config_and_filters")

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
            perf.start("video:decode_frames")

            os.makedirs(temp_dir, exist_ok=True)

            # Set result save paths (kept for compatibility with coord pickle path)
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)

            # Set output video path
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)

            # Decode frames in-memory instead of extracting to PNGs
            if get_file_type(video_path) == "video":
                cap = cv2.VideoCapture(video_path)
                fps = get_video_fps(video_path)
                input_img_list = []  # will hold in-memory frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    input_img_list.append(frame)
                cap.release()
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
            perf.end("video:decode_frames")


            # Extract audio features
            perf.start("audio:preprocess")

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
            perf.end("audio:preprocess", extra_note=f"secs={librosa_length/16000.0:.2f}")
            acc_media_seconds += (librosa_length / 16000.0)

            # Preprocess input images
            perf.start("preprocessing:coords_and_frames")

            pitch_filtered_count = 0  # Initialize pitch filtering counter
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                # Use in-memory frames directly when provided, else read from paths
                if len(input_img_list) > 0 and not isinstance(input_img_list[0], (str, bytes, os.PathLike)):
                    frame_list = input_img_list
                else:
                    frame_list = read_imgs(input_img_list)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, frame_list, pitch_filtered_count = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)

            perf.end("preprocessing:coords_and_frames", extra_note=f"frames={len(frame_list)}")

            perf.start("inference:vae_encoding")

            print(f"Number of frames: {len(frame_list)}")

            # Batch VAE encode: accumulate valid crops and encode in chunks
            input_latent_list = []
            crops = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                # Skip frames with invalid bbox dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                if args.version == "v15":
                    y2 = min(y2 + args.extra_margin, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crops.append(crop_frame)

            vae_bs = getattr(args, "vae_batch_size", 32)
            for i in range(0, len(crops), vae_bs):
                batch_imgs = [cv2.resize(c, (256, 256), interpolation=cv2.INTER_LANCZOS4) for c in crops[i:i+vae_bs]]
                lat_list = vae.get_latents_for_unet_batch(batch_imgs)
                input_latent_list.extend(lat_list)

            perf.end("inference:vae_encoding", extra_note=f"latents={len(input_latent_list)}")


            # Smooth first and last frames
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Get video length from audio chunks
            video_num = len(whisper_chunks)

            # Handle case where no valid faces were detected in any frame
            if len(input_latent_list) == 0:
                print("No valid face detections found. Using passthrough mode - streaming original frames.")
                # Stream original frames directly to ffmpeg with audio
                perf.start("postprocessing:stream_encode")
                h, w = frame_list_cycle[0].shape[:2]
                ffmpeg_bin = "ffmpeg"
                try:
                    if args.ffmpeg_path and os.path.exists(args.ffmpeg_path):
                        if os.path.isdir(args.ffmpeg_path):
                            cand = os.path.join(args.ffmpeg_path, "ffmpeg")
                            if os.path.exists(cand):
                                ffmpeg_bin = cand
                        elif os.path.isfile(args.ffmpeg_path):
                            ffmpeg_bin = args.ffmpeg_path
                except Exception:
                    pass
                ffmpeg_cmd = [
                    ffmpeg_bin, "-y", "-v", "warning",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                    "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", getattr(args, "ffmpeg_preset", "veryfast"), "-crf", "18", "-vf", "format=yuv420p",
                    "-shortest",
                    output_vid_name,
                ]
                if getattr(args, "ffmpeg_threads", None):
                    ffmpeg_cmd = ffmpeg_cmd[:-1] + ["-threads", str(args.ffmpeg_threads), ffmpeg_cmd[-1]]
                proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                for i in range(video_num):
                    ori_frame = frame_list_cycle[i % len(frame_list_cycle)]
                    try:
                        proc.stdin.write(ori_frame.astype(np.uint8).tobytes())
                    except BrokenPipeError:
                        break
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
                perf.end("postprocessing:stream_encode")
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

                perf.start("inference:unet_forward")

                res_frame_list = []
                total = int(np.ceil(float(video_num) / batch_size))

                # Execute inference
                with torch.inference_mode():
                    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                        audio_feature_batch = pe(whisper_batch)
                        latent_batch = latent_batch.to(dtype=unet.model.dtype)

                        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                        recon = vae.decode_latents(pred_latents)
                        for res_frame in recon:
                            res_frame_list.append(res_frame)

                perf.end("inference:unet_forward", extra_note=f"frames={len(res_frame_list)}")

                # Pad generated images to original video size and stream-encode with audio in one pass
                print("Padding generated images to original video size")

                # Prepare ffmpeg streaming process
                perf.start("postprocessing:stream_encode")
                h, w = frame_list_cycle[0].shape[:2]
                ffmpeg_bin = "ffmpeg"
                try:
                    if args.ffmpeg_path and os.path.exists(args.ffmpeg_path):
                        if os.path.isdir(args.ffmpeg_path):
                            cand = os.path.join(args.ffmpeg_path, "ffmpeg")
                            if os.path.exists(cand):
                                ffmpeg_bin = cand
                        elif os.path.isfile(args.ffmpeg_path):
                            ffmpeg_bin = args.ffmpeg_path
                except Exception:
                    pass
                ffmpeg_cmd = [
                    ffmpeg_bin, "-y", "-v", "warning",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                    "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", getattr(args, "ffmpeg_preset", "veryfast"), "-crf", "18", "-vf", "format=yuv420p",
                    "-shortest",
                    output_vid_name,
                ]
                if getattr(args, "ffmpeg_threads", None):
                    ffmpeg_cmd = ffmpeg_cmd[:-1] + ["-threads", str(args.ffmpeg_threads), ffmpeg_cmd[-1]]
                proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


                res_frame_idx = 0  # Track index in res_frame_list separately

                for i in range(video_num):
                    bbox = coord_list_cycle[i % (len(coord_list_cycle))]
                    ori_frame = frame_list_cycle[i % (len(frame_list_cycle))].copy()
                    x1, y1, x2, y2 = bbox

                    out_frame = ori_frame  # default

                    # Handle frames with no face detection (placeholder or invalid bbox)
                    if not (bbox == coord_placeholder or x2 <= x1 or y2 <= y1):
                        if res_frame_idx < len(res_frame_list):
                            res_frame = res_frame_list[res_frame_idx]
                            res_frame_idx += 1

                            if args.version == "v15":
                                y2 = min(y2 + args.extra_margin, ori_frame.shape[0])

                            # Optional 'no lip' presence check for bypass
                            do_bypass = False
                            if getattr(args, 'enable_no_lip_bypass', False):
                                try:
                                    crop_for_parse = ori_frame[y1:y2, x1:x2]
                                    lip_mask_img = fp(Image.fromarray(crop_for_parse), mode="lips_only")
                                    lip_mask_np = np.array(lip_mask_img)
                                    lip_pixels = int(np.count_nonzero(lip_mask_np))
                                    lip_frac = lip_pixels / float(lip_mask_np.size) if lip_mask_np.size > 0 else 0.0
                                    if lip_pixels == 0 or lip_frac < getattr(args, 'no_lip_min_frac', 0.0015):
                                        do_bypass = True
                                except Exception:
                                    pass

                            if not do_bypass:
                                try:
                                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                                    # Merge results with version-specific parameters
                                    if args.version == "v15":
                                        out_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                                    else:
                                        out_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                                except Exception:
                                    out_frame = ori_frame
                            else:
                                no_lip_bypass_count += 1
                                out_frame = ori_frame

                    # Write frame to ffmpeg stdin
                    try:
                        proc.stdin.write(out_frame.astype(np.uint8).tobytes())
                    except BrokenPipeError:
                        break

                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
                perf.end("postprocessing:stream_encode")

                # Report pitch filtering results
                if args.enable_pitch_filter:
                    print(f"Pitch filtering: {pitch_filtered_count} frames skipped due to pitch detection criteria")

                # Report no-lip bypass results
                if getattr(args, 'enable_no_lip_bypass', False) and no_lip_bypass_count > 0:
                    print(f"No-lip bypass: {no_lip_bypass_count} frames used original frames due to insufficient lip detection")

            # Clean up temporary files (coordinate pickle optional)
            if not args.saved_coord:
                try:
                    os.remove(crop_coord_save_path)
                except Exception:
                    pass

            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)

    # Final performance report
    total_elapsed = time.perf_counter() - total_start
    print(f"[PROF] Total wall time: {total_elapsed:.3f}s")
    perf.report(total_media_seconds=acc_media_seconds)

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
    parser.add_argument("--ffmpeg_preset", type=str, default="veryfast", help="FFmpeg x264 preset (e.g., ultrafast..veryslow)")

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
    parser.add_argument("--vae_batch_size", type=int, default=32, help="Batch size for VAE encoding")

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
    # Parallelization/coordinator flags
    parser.add_argument("--num_threads", type=int, default=1, help="Number of parallel inference processes to run. 1 = normal single-process mode.")
    parser.add_argument("--cpu_threads_per_job", type=int, default=None, help="Override CPU threads for BLAS/OpenMP per job. If not set, coordinator estimates.")
    parser.add_argument("--ffmpeg_threads", type=int, default=None, help="Threads to pass to ffmpeg (-threads). If unset, ffmpeg decides.")
    parser.add_argument("--stagger_start_secs", type=int, default=0, help="Stagger start between child jobs to avoid VRAM spikes.")
    parser.add_argument("--oom_retry_factor", type=float, default=0.75, help="Scale factor for batch sizes on OOM retry (coordinator).")

    parser.add_argument("--pitch_down_threshold", type=float, default=30.0, help="Pitch angle threshold for filtering (degrees downward)")
    parser.add_argument("--pitch_up_threshold", type=float, default=20.0, help="Pitch angle threshold for exiting filter (degrees downward)")
    args = parser.parse_args()
    main(args)

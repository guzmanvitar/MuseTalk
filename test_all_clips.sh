#!/bin/bash

# Test script for comprehensive test clips with no face/occlusion cases
# Usage: ./test_all_clips.sh [v1.0|v1.5] [normal|realtime]

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <mode>"
    echo "Example: $0 v1.5 normal"
    exit 1
fi

version=$1
mode=$2

# Validate mode
if [ "$mode" != "normal" ] && [ "$mode" != "realtime" ]; then
    echo "Invalid mode specified. Please use 'normal' or 'realtime'."
    exit 1
fi

# Define the model paths based on the version
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
    unet_config="$model_dir/musetalk.json"
    version_arg="v15"
else
    echo "Invalid version specified. Please use v1.0 or v1.5."
    exit 1
fi

# Set paths based on mode
if [ "$mode" = "normal" ]; then
    config_path="./configs/inference/test_comprehensive.yaml"
    result_dir="./results/test_comprehensive"
else
    config_path="./configs/inference/realtime.yaml"
    result_dir="./results/realtime_comprehensive"
fi

# Create results directory
mkdir -p "$result_dir"

# Base command arguments
cmd_args="--inference_config $config_path \
    --result_dir $result_dir \
    --unet_model_path $unet_model_path \
    --unet_config $unet_config \
    --version $version_arg \
    --enable_no_lip_bypass \
    --enable_pose2d_filter \
    --pose2d_vpi_thr 1.10 \
    --pose2d_lfc_thr 0.19 \
    --ffmpeg_preset veryfast \
    --vae_batch_size 12 \
    --batch_size 4 \
    --num_threads 4 \
    --stagger_start_secs 30 \
    --ffmpeg_threads 2 \
    --cpu_threads_per_job 4 \
    --use_float16 \
    --enable_foreground_preserve \
    --foreground_preserve_dilate_px 14 \
    --foreground_preserve_roi_scale 4.0 \
    --foreground_preserve_temporal_smooth 8 \
    --foreground_preserve_min_det_conf 0.30 \
    --foreground_preserve_min_track_conf 0.30"

echo "Testing comprehensive clips with MuseTalk $version in $mode mode"
echo "Config: $config_path"
echo "Results will be saved to: $result_dir"
echo ""

# Set script name based on mode
if [ "$mode" = "normal" ]; then
    script_name="scripts.inference"
else
    script_name="scripts.realtime_inference"
fi

# Run inference using the virtual environment
source MuseTalk_env/bin/activate && python -m $script_name $cmd_args

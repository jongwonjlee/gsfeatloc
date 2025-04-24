# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : run_gsfeatloc_sensitivity_analysis.sh
# Description   : This is the bash script for the sensitivity analysis of GSFeatLoc on rotation and translation deviations 
#                 of the initial pose from the ground truth, on the Lego scene of the Blender dataset.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0

# Function to process datasets
run_gsfeatloc_sensitivity_analysis() {
    local scene_path=$1
    local model_path=$2
    local output_path=$3

    # Loop over all subdirectories in the dataset path
    for s in "$scene_path"/lego; do # FIXME: Change this to loop over all subdirectories
    if [ -d "$s" ]; then # Ensure it's a directory
        # Get the name of the subdirectory
        seq=$(basename "$s")

        # Create the model directory if it does not exist
        echo "Output directory: $output_path/$seq/sensitivity-analysis-step-10"
        mkdir -p "$output_path/$seq/sensitivity-analysis-step-10"
        
        for rot in 33.84 30.45 27.07 23.68 20.30 16.92 13.53 10.15 6.77 3.38 0.00; do
            trs=0.00

            rot_str=$(printf "%.2f" "$rot" | sed 's/\./p/')
            trs_str=$(printf "%.2f" "$trs" | sed 's/\./p/')

            # Run the GSFeatLoc CLI for each combination of rotation and translation
            echo "Running GSFeatLoc with rotation: $rot, translation: $trs"
            python ../gsfeatloc_cli.py \
                --scene_path "$scene_path/$seq" \
                --model_path "$model_path/$seq/checkpoint-30000" \
                --output_path "$output_path/$seq/sensitivity-analysis-step-10" \
                --output_filename "pose_dr_${rot_str}_dt_${trs_str}.json" \
                --delta_rot "$rot" \
                --delta_trs "$trs" \
                --axis_mode 'x' \
                --magnitude_mode 'fixed' | tee "$output_path/$seq/sensitivity-analysis-step-10/pose_dr_${rot_str}_dt_${trs_str}.log"
        done

        for trs in 2.44 2.20 1.95 1.71 1.46 1.22 0.98 0.73 0.49 0.24; do
            rot=0.00

            rot_str=$(printf "%.2f" "$rot" | sed 's/\./p/')
            trs_str=$(printf "%.2f" "$trs" | sed 's/\./p/')

            # Run the gsfeatloc CLI for each combination of rotation and translation
            echo "Running gsfeatloc with rotation: $rot, translation: $trs"
            python ../gsfeatloc_cli.py \
                --scene_path "$scene_path/$seq" \
                --model_path "$model_path/$seq/checkpoint-30000" \
                --output_path "$output_path/$seq/sensitivity-analysis-step-10" \
                --output_filename "pose_dr_${rot_str}_dt_${trs_str}.json" \
                --delta_rot "$rot" \
                --delta_trs "$trs" \
                --axis_mode 'x' \
                --magnitude_mode 'fixed' | tee "$output_path/$seq/sensitivity-analysis-step-10/pose_dr_${rot_str}_dt_${trs_str}.log"
        done
    fi
    done
}

# Run blender dataset
run_gsfeatloc_sensitivity_analysis "/home/jongwonlee/datasets/nerfbaselines/blender" \
                                   "/home/jongwonlee/models/gsplat/blender" \
                                   "/home/jongwonlee/output/gsfeatloc/blender"
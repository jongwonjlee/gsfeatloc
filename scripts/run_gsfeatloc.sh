# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : run_gsfeatloc.sh
# Description   : This is the bash script to execute GSFeatLoc on various datasets with randomized initial poses, 
#                 ensuring 95% probability of keeping the object in view through specified rotation and translation magnitudes from the ground-truth.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0

# Function to process datasets
run_gsfeatloc() {
    local dataset_path=$1
    local model_path=$2
    local output_path=$3
    local scene=$4
    local rot=$5
    local trs=$6

    # Check if $dataset_path/$scene exists
    if [ ! -d "$dataset_path/$scene" ]; then
        echo "Error: Scene path $dataset_path/$scene does not exist."
        return 1
    fi

    # Check if $model_path/$scene exists
    if [ ! -d "$model_path/$scene" ]; then
        echo "Error: Model path $model_path/$scene does not exist."
        return 1
    fi

    echo "Running GSFeatLoc for scene: $scene"
    # echo "Applying perturbations: Rotation = $rot degrees, Translation = $trs meters"

    # Create the output directory if it does not exist
    echo "Output directory: $output_path/$scene"
    mkdir -p "$output_path/$scene"

    # Run the GSFeatLoc CLI for each combination of rotation and translation
    echo "python ../gsfeatloc_cli.py \
        --scene_path $dataset_path/$scene \
        --model_path $model_path/$scene/checkpoint-30000 \
        --output_path $output_path/$scene \
        --output_filename results.json \
        --delta_rot $rot \
        --delta_trs $trs \
        --axis_mode 'random' \
        --magnitude_mode 'gaussian' | tee $output_path/$scene/results.log"

    # Comment out the following line to prevent actual execution
    python ../gsfeatloc_cli.py \
        --scene_path $dataset_path/$scene \
        --model_path $model_path/$scene/checkpoint-30000 \
        --output_path $output_path/$scene \
        --output_filename results.json \
        --delta_rot $rot \
        --delta_trs $trs \
        --axis_mode 'random' \
        --magnitude_mode 'gaussian' | tee $output_path/$scene/results.log
}

# Define datasets and their scenes with corresponding delta_rot and delta_trs values
declare -A blender_scenes=(
    ["chair"]="12.10 0.87"
    ["drums"]="12.10 0.87"
    ["ficus"]="12.10 0.87"
    ["hotdog"]="12.10 0.87"
    ["lego"]="12.10 0.87"
    ["materials"]="12.10 0.87"
    ["mic"]="12.10 0.87"
    ["ship"]="12.10 0.87"
)

declare -A mipnerf360_scenes=(
    ["bicycle"]="15.04 1.20"
    ["bonsai"]="14.26 1.13"
    ["counter"]="14.32 1.15"
    ["flowers"]="15.80 1.22"
    ["garden"]="17.18 1.36"
    ["kitchen"]="14.23 1.10"
    ["room"]="14.37 1.44"
    ["stump"]="15.32 1.20"
    ["treehill"]="16.11 1.32"
)

declare -A tanksandtemples_scenes=(
    ["auditorium"]="19.43 2.07"
    ["barn"]="19.39 1.74"
    ["church"]="19.41 2.30"
    ["courtroom"]="19.42 2.03"
    ["francis"]="19.45 1.83"
    ["ignatius"]="19.39 1.99"
    ["m60"]="16.78 1.59"
    ["museum"]="19.44 1.73"
    ["panther"]="16.78 1.39"
    ["temple"]="19.39 1.98"
    ["truck"]="19.40 1.95"
    ["ballroom"]="19.41 2.14"
    ["caterpillar"]="19.41 1.97"
    ["courthouse"]="19.37 2.26"
    ["family"]="19.38 1.80"
    ["horse"]="19.41 2.20"
    ["lighthouse"]="16.78 1.68"
    ["meetingroom"]="19.41 1.95"
    ["palace"]="16.13 1.95"
    ["playground"]="14.48 1.28"
    ["train"]="19.43 2.04"
)

# Function to run gsfeatloc for all scenes in a dataset
run_dataset() {
    local dataset_path=$1
    local model_path=$2
    local output_path=$3
    declare -n scenes=$4

    for seq in "${!scenes[@]}"; do
        read rot trs <<< "${scenes[$scene]}"
        run_gsfeatloc "$dataset_path" "$model_path" "$output_path" "$scene" "$rot" "$trs"
        # break
    done
}

# Run for Blender dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/blender" \
            "/home/jongwonlee/models/gsplat/blender" \
            "/home/jongwonlee/output/gsfeatloc/blender" \
            blender_scenes

# Run for MipNeRF360 dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/mipnerf360" \
            "/home/jongwonlee/models/gsplat/mipnerf360" \
            "/home/jongwonlee/output/gsfeatloc/mipnerf360" \
            mipnerf360_scenes

# # Run for Tanks and Temples dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/tanksandtemples" \
            "/home/jongwonlee/models/gsplat/tanksandtemples" \
            "/home/jongwonlee/output/gsfeatloc/tanksandtemples" \
            tanksandtemples_scenes
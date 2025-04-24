# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : run_gsfeatloc_6dgs.sh
# Description   : This is the bash script to execute GSFeatLoc on various datasets with initial poses provided by 6DGS.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0

# Function to process datasets
run_gsfeatloc_6dgs() {
    local scene_path=$1
    local model_path=$2
    local output_path=$3
    local sdgs_path=$4
    local seq=$5

    # Check if $scene_path/$seq exists
    if [ ! -d "$scene_path/$seq" ]; then
        echo "Error: Scene path $scene_path/$seq does not exist."
        return 1
    fi

    # Check if $model_path/$seq exists
    if [ ! -d "$model_path/$seq" ]; then
        echo "Error: Model path $model_path/$seq does not exist."
        return 1
    fi

    echo "Running GSFeatLoc for scene: $seq"
    echo "Applying perturbations: Rotation = $rot degrees, Translation = $trs meters"

    # Create the output directory if it does not exist
    echo "Output directory: $output_path/$seq"
    mkdir -p "$output_path/$seq"

    # Run the GSFeatLoc CLI for each combination of rotation and translation
    echo "python ../gsfeatloc_cli.py \
        --scene_path $scene_path/$seq \
        --model_path $model_path/$seq/checkpoint-30000 \
        --output_path $output_path/$seq \
        --sdgs_path $sdgs_path/$seq \
        --output_filename results-6DGS.json | tee $output_path/$seq/results-6DGS.log"

    # Comment out the following line to prevent actual execution
    python ../gsfeatloc_cli.py \
        --scene_path $scene_path/$seq \
        --model_path $model_path/$seq/checkpoint-30000 \
        --output_path $output_path/$seq \
        --sdgs_path $sdgs_path/$seq \
        --output_filename results-6DGS.json | tee $output_path/$seq/results-6DGS.log
}

# Define datasets and their sequences
blender_sequences=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
mipnerf360_sequences=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
tanksandtemples_sequences=("auditorium" "barn" "church" "courtroom" "francis" "ignatius" "m60" "museum" "panther" "temple" "truck" "ballroom" "caterpillar" "courthouse" "family" "horse" "lighthouse" "meetingroom" "palace" "playground" "train")


# Function to run gsfeatloc for all sequences in a dataset
run_dataset() {
    local scene_path=$1
    local model_path=$2
    local output_path=$3
    local sdgs_path=$4
    declare -n sequences=$5

    for seq in "${sequences[@]}"; do
        run_gsfeatloc_6dgs "$scene_path" "$model_path" "$output_path" "$sdgs_path" "$seq"
        # break
    done
}

# Run for Blender dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/blender" \
            "/home/jongwonlee/models/gsplat/blender" \
            "/home/jongwonlee/output/gsfeatloc/blender" \
            "/home/jongwonlee/output/6dgs/blender" \
            blender_sequences

# Run for MipNeRF360 dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/mipnerf360" \
            "/home/jongwonlee/models/gsplat/mipnerf360" \
            "/home/jongwonlee/output/gsfeatloc/mipnerf360" \
            "/home/jongwonlee/output/6dgs/mipnerf360" \
            mipnerf360_sequences

# Run for Tanks and Temples dataset
run_dataset "/home/jongwonlee/datasets/nerfbaselines/tanksandtemples" \
            "/home/jongwonlee/models/gsplat/tanksandtemples" \
            "/home/jongwonlee/output/gsfeatloc/tanksandtemples" \
            "/home/jongwonlee/output/6dgs/tanksandtemples" \
            tanksandtemples_sequences
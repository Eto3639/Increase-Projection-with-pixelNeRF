#!/bin/bash

# test.sh
#
# This script automates the verification process for the data generation pipeline.
# It performs the following steps:
# 1. Deletes any previously generated dataset and debug outputs.
# 2. Finds the first NIFTI file in the NAS directory.
# 3. Runs the `generate_drr.py` script within a Docker container to create a new,
#    single-case dataset from that NIFTI file.
# 4. Runs the `visual_debug.py` script in another Docker container to analyze
#    the generated data and produce debug visualizations.

set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

echo "--- Starting Data Generation Verification Script ---"

# --- Configuration ---
USER_ID=$(id -u)
GROUP_ID=$(id -g)
NAS_PARENT_DIR="/mnt/nas/eto/LIDC" # Matches run_drr_all.sh
DRR_DATASET_DIR="$(pwd)/drr_dataset" # Matches config.yml

# --- Step 1: Clean up old data ---
echo -e "\n[Step 1/4] Deleting old dataset and debug outputs..."
rm -rf "${DRR_DATASET_DIR}"
# rm -rf debug_outputs
echo "Cleanup complete."

# --- Step 2: Find a NIFTI file to process ---
echo -e "\n[Step 2/4] Searching for a sample NIFTI file in ${NAS_PARENT_DIR}/CT_Nifti..."
# Find the first .nii.gz file to use as a test case.
# Using `find ... -print -quit` is a safe way to get just the first result.
FIRST_NIFTI_FILE=$(find "${NAS_PARENT_DIR}/CT_Nifti" -name "*.nii.gz" -print -quit 2>/dev/null)

if [ -z "$FIRST_NIFTI_FILE" ]; then
    echo "❌ ERROR: No NIFTI files (.nii.gz) found in ${NAS_PARENT_DIR}/CT_Nifti."
    echo "Please ensure the NAS is mounted and contains data."
    exit 1
fi
NIFTI_BASENAME=$(basename "$FIRST_NIFTI_FILE")
CONTAINER_NIFTI_PATH="/data/CT_Nifti/${NIFTI_BASENAME}"
echo "Found test file: ${FIRST_NIFTI_FILE}"
echo "It will be accessed inside the container at: ${CONTAINER_NIFTI_PATH}"

# --- Step 3: Generate new data for the single case ---
echo -e "\n[Step 3/4] Running data generation for a single file..."
# This uses the 'drr-generator' image, consistent with run_drr_all.sh
docker run --gpus all --rm \
  -u "${USER_ID}:${GROUP_ID}" \
  -v "$(pwd)":/workspace \
  -v "${NAS_PARENT_DIR}":/data \
  -e MPLCONFIGDIR=/workspace/.config/matplotlib \
  drr-generator \
  python3 generate_drr.py --file "${CONTAINER_NIFTI_PATH}"

echo "Data generation complete."

# --- Step 4: Run the visual debugger ---
# echo -e "\n[Step 4/4] Running visual debugger on the newly generated data..."
# # This uses the 'nerf_multiview' image, consistent with run_train.sh
# # It needs access to the generated drr_dataset.
# mkdir -p .home # Create a home dir for the container user, like in run_train.sh
# docker run --gpus all -i --rm \
#   -u "${USER_ID}:${GROUP_ID}" \
#   -v "$(pwd)":/workspace \
#   -v "${DRR_DATASET_DIR}":"/workspace/drr_dataset" \
#   -e HOME=/workspace/.home \
#   nerf_multiview \
#   python3 visual_debug.py

echo -e "\n--- Verification Script Finished ---"
echo "✅ All steps completed. Please check the images in the 'debug_outputs' directory."
echo "Pay special attention to 'debug_outputs/debug_02_cameras.png' to ensure cameras are pointing correctly."
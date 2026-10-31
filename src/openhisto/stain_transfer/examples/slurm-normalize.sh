#!/usr/bin/env bash
#SBATCH --job-name=stain-transfer
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9
#SBATCH --output=stain-transfer-%A_%a.log
set -euo pipefail
# Each line in jobs.txt is one independent normalization job JSON path.
# Adjust the array range to match the number of lines. Use separate output paths.
mapfile -t jobs < "${STAIN_JOB_LIST:-jobs.txt}"
job="${jobs[$SLURM_ARRAY_TASK_ID]:?No job for this array index}"
apptainer run --bind "${STAIN_DATA_ROOT:?Set STAIN_DATA_ROOT}" "${STAIN_SIF:-stain-transfer.sif}" normalize --job "$job"
# For GAN inference request a GPU in your scheduler and add --nv above.

#!/bin/bash

#SBATCH --job-name=asr-mal

#SBATCH --output=results.txt

#SBATCH --gres=gpu:1       # Request 1 GPU (adjust if needed)

#SBATCH --ntasks=4         # Use 4 CPU tasks (for data loading)

#SBATCH --mem=32G          # Request 32GB total RAM

#SBATCH --account=masters

#SBATCH --partition=regular

hostname
python3 tamal_cpt.py

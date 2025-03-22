#!/bin/bash

#SBATCH --job-name=asr-mal

#SBATCH --output=results.txt

#SBATCH --gres=gpu:3

#SBATCH --ntasks=10

#SBATCH --mem-per-cpu=1024

hostname
python3 speech_wav2mal_new.py

#!/bin/sh 
#SBATCH -J gpu_job 
#SBATCH -c 6 
#SBATCH -p gpu-shared
#SBATCH --gres=gpu
#SBATCH -t 00:05:00

#Blocks of 256 threads
python3 benchmark-gpu-kaggle.py


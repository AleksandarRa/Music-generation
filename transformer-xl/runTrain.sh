#!/bin/bash
#
#SBATCH --job-name=train_music_transformer
#SBATCH --output=logs/out_train.txt
#SBATCH -e logs/error_train.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --cpus-per-task=1

sleep 5
/home2/aradovic/miniconda3/envs/rtxenv/bin/python /home2/aradovic/Music-generation/transformer-xl/train.py

exit

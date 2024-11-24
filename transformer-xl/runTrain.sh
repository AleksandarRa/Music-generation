#!/bin/bash
#
#SBATCH --job-name=train_music_transformer
#SBATCH --output=logs/out_train.txt
#SBATCH -e logs/error_train.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-20:00
#SBATCH --mem-per-cpu=2000

sleep 5
/home2/aradovic/miniconda3/envs/a6000env/bin/python /home2/aradovic/Music-generation/transformer-xl/train.py

exit

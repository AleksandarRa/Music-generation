#!/bin/bash
#
#SBATCH --job-name=train_music_LSTM
#SBATCH --output=out_train_LSTM.txt
#SBATCH -e error_train_LSTM.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-20:00
#SBATCH --mem-per-cpu=2000

sleep 5
/home2/aradovic/miniconda3/envs/myenv/bin/python /home2/aradovic/Music-generation/transformer-xl/train_LSTM.py

exit

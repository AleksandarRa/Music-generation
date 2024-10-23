#!/bin/bash
#
#SBATCH --job-name=generateMusic
#SBATCH --output=out_generate_transformerXL.txt
#SBATCH -e error_generate_transformerXL.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-10:00
#SBATCH --mem-per-cpu=2000

sleep 5
/home2/aradovic/miniconda3/envs/myenv/bin/python /home2/aradovic/Music-generation/transformer-xl/generate_music_transformerXL.py

exit

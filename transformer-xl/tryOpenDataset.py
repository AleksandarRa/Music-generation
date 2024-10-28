from midi_parser import MIDI_parser
import config_music as config
from utils import shuffle_ragged_2d, inputs_to_labels, get_quant_time
import numpy as np
import tensorflow as tf
import argparse
import os
import pathlib

idx_to_time = get_quant_time()

midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

print('Creating dataset')
dataset = midi_parser.get_tf_dataset(
            file_directory="npz_temp", batch_size=config.batch_size)

print("Finish")

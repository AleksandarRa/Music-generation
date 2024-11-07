import pathlib
import numpy as np
import csv
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config

# Initialize MIDI parser and get list of npz files
idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
npz_filenames = list(pathlib.Path("data/npz").rglob('*.npz'))

# Prepare a list to store file sizes and names
file_info = []

# Loop through each file and retrieve its size and name
for filename in npz_filenames:
    sound, _ = midi_parser.load_features(filename)
    size = sound.size  # Assuming sound.size gives the correct size measurement
    file_info.append((size, filename.name))  # Store size and filename

# Sort the file information by file size in descending order
file_info.sort(reverse=True, key=lambda x: x[0])

# Write the sorted information to a CSV file
with open('logs/npz_sizes.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Filesize', 'Filename'])  # Write header
    writer.writerows(file_info)  # Write each file's info

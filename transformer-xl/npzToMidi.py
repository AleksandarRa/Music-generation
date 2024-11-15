import os
import pathlib
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config
import os
import numpy as np

# Initialize MIDI parser and get list of npz files
idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
npz_filenames = list(pathlib.Path("data/npz_temp").rglob('*.npz'))
file_names_without_extension = [file.stem for file in npz_filenames]
file_names_array = np.array(file_names_without_extension)

midi_filenames = [f + '.midi' for f in file_names_array]

sounds, deltas = zip(*[midi_parser.load_features(filename)
                       for filename in npz_filenames])

midi_list = [midi_parser.features_to_midi(
    sound, delta) for sound, delta in zip(sounds, deltas)]

for midi, filename in zip(midi_list, midi_filenames):
    midi.save("data/midi_temp/"+filename)

import pathlib
import numpy as np
import csv
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config

# Initialize MIDI parser and get list of npz files
idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
npz_filenames = list(pathlib.Path("data/npz_temp").rglob('*.npz'))

midi_filenames = [str(i) for i in range(1, len(npz_filenames)+1)]

midi_filenames = [f + '.midi' for f in midi_filenames]

sounds, deltas = zip(*[midi_parser.load_features(filename)
                       for filename in npz_filenames])

midi_list = [midi_parser.features_to_midi(
    sound, delta) for sound, delta in zip(sounds, deltas)]

for midi, filename in zip(midi_list, midi_filenames):
    midi.save("data/midi_temp/"+filename)

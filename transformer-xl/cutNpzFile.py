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
for i in range (1,5):
    npz_filenames = list(pathlib.Path("data/analysedData/"+str(i)+"approach/npz/fullSong/").rglob('*.npz'))
    file_names_without_extension = [file.stem for file in npz_filenames]
    file_names_array = np.array(file_names_without_extension)

    start = 1500
    end = 3000
    midi_filenames = [f +'npz_'+ str(start) + '-'+str(end) + '.npz' for f in file_names_array]

    sounds, deltas = zip(*[midi_parser.load_features(filename)
                           for filename in npz_filenames])

    sounds = tuple(sound[start:end] for sound in sounds)
    deltas = tuple(delta[start:end] for delta in deltas)

    for z in range (0,4):
        np.savez("data/analysedData/"+str(i)+"approach/npz/"+str(start)+"-"+str(end)+"/"+midi_filenames[z], sounds=sounds[z], deltas=deltas[z])

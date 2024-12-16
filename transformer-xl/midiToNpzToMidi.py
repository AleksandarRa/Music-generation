import os
import pathlib
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config
import os
import numpy as np

APPROACH=4
FILENAMES = ['138', '255', '341', '346']

def midiToNpz(midi_dir, filename, dst_path):

    ext_list = ['*.midi', '*.mid']
    midi_filenames = []
    for ext in ext_list:
        ext_filenames = pathlib.Path(midi_dir).rglob(ext)
        ext_filenames = list(map(lambda x: str(x), ext_filenames))
        midi_filenames += ext_filenames

    print(f'Found {len(midi_filenames)} midi files')
    assert len(midi_filenames) > 0

    idx_to_time = get_quant_time()
    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

    midi_parser.preprocess_dataset(src_filenames=midi_filenames,
                                   dst_dir=dst_path, batch_size=1, dst_filenames=None)

def npzToMidi(npz_path, midi_path):
    # Initialize MIDI parser and get list of npz files
    idx_to_time = get_quant_time()
    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
    npz_filenames = list(pathlib.Path(npz_path).rglob('*.npz'))
    file_names_without_extension = [file.stem for file in npz_filenames]
    file_names_array = np.array(file_names_without_extension)

    midi_filenames = [f + '.midi' for f in file_names_array]

    sounds, deltas = zip(*[midi_parser.load_features(filename)
                           for filename in npz_filenames])

    midi_list = [midi_parser.features_to_midi(
        sound, delta) for sound, delta in zip(sounds, deltas)]

    for midi, filename in zip(midi_list, midi_filenames):
        midi.save(midi_path+filename)

for FILENAME in FILENAMES:
    #midiToNpz(f'data/analysedData/{APPROACH}approach/midi/fullSong', FILENAME, f'data/analysedData/{APPROACH}approach/preprocessingTest/1_npz/{FILENAME}.npz')
    #npzToMidi(f'data/analysedData/{APPROACH}approach/preprocessingTest/1_npz', f'data/analysedData/{APPROACH}approach/preprocessingTest/2_midi/')
    midiToNpz(f'data/analysedData/{APPROACH}approach/preprocessingTest/2_midi/', FILENAME,
          f'data/analysedData/{APPROACH}approach/preprocessingTest/3_npz/{FILENAME}.npz')

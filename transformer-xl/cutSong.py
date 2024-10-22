import pathlib

import config_music as config
from midi_parser import MIDI_parser
from utils import get_quant_time, generate_midis, saveToBuffer
import numpy as np

npz_filenames = list(pathlib.Path('npz_music_maestro').rglob('0.npz'))
filenames = np.random.choice(
    npz_filenames, 1, replace=False)

idx_to_time = get_quant_time()
parser = MIDI_parser.build_from_config(config, idx_to_time)
sounds, deltas = zip(*[parser.load_features(filename)
                       for filename in filenames])

sounds = np.array([sound for sound in sounds])
deltas = np.array([delta for delta in deltas])

maxlen = sounds.shape[1]
seq_len = int(maxlen / 4) - 1

saveToBuffer((sounds, deltas), parser, 'test/full_song.midi')

sound = sounds[:, 0:seq_len]
delta = deltas[:, 0:seq_len]
saveToBuffer((sound, delta), parser, 'test/part_1.midi')
sound = sounds[:, seq_len:2*seq_len]
delta = deltas[:, seq_len:2*seq_len]
saveToBuffer((sound, delta), parser, 'test/part_2.midi')
sound = sounds[:, seq_len:3*seq_len]
delta = deltas[:, seq_len:3*seq_len]
saveToBuffer((sound, delta), parser, 'test/part_3.midi')
sound = sounds[:, seq_len:4*seq_len]
delta = deltas[:, seq_len:4*seq_len]
saveToBuffer((sound, delta), parser, 'test/part_4.midi')

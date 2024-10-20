import pickle

import pretty_midi

from utils import get_quant_time
import config_music as config
from midi_parser import MIDI_parser
import mido
import numpy as np
from scipy.sparse import coo_matrix

# Replace with your file path
file_path = 'datasets/classical/archive/music/music.pk'

# Open and load the pickle file
with open(file_path, 'rb') as f:
    allData = pickle.load(f)

# Now 'data' contains the loaded object
# print(allData)

idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
# Assuming data is a dictionary with MIDI bytes
j=0
for midi_data in allData:
    midi_name = midi_data['name']
    midi_composer = midi_data['composer']
    midi_piece = midi_data['piece']

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an instrument
    instrument = pretty_midi.Instrument(program=0)  # 0 for acoustic grand piano

    # The `midi_piece` contains the note, time, and velocity data.
    # You will have to interpret the sparse matrix structure.
    # Assuming that 'row' corresponds to time steps, 'col' corresponds to notes,
    # and 'data' corresponds to velocities.
    row = midi_piece.row
    col = midi_piece.col
    data = midi_piece.data

    # Loop through the sparse matrix to add notes to the instrument
    for i in range(len(data)):
        start_time = row[i] * 0.1  # Adjust time scaling factor if necessary
        end_time = start_time + 0.1  # Adjust duration scaling factor if necessary
        pitch = int(col[i])  # MIDI note number
        if data[i] > 1:
            data[i] = 1
        velocity = int(data[i] * 127)  # Scale velocity to 0-127 range

        # Create a note and add it to the instrument
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write the MIDI file
    midi.write('output_midi_file_'+str(j)+'.mid')
    j+=1
    if j == 10:
        exit()

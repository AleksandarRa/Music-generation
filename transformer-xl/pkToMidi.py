import pickle
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

# Now 'data' contains the loaded object (it could be a dictionary, list, etc.)

idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)
# Assuming data is a dictionary with MIDI bytes
for midi_data in allData:
    midi_name = midi_data['name']
    midi_composer = midi_data['composer']
    midi_piece = midi_data['piece']

    # Convert the sparse matrix to a dense numpy array
    midi_matrix = midi_piece.toarray()

    # Create a new MIDI file
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Set initial tempo (adjust this value as needed)
    tempo = mido.bpm2tempo(120)  # Default 120 BPM
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    time_step_array = []
    velocity_array = []
    # Loop over time steps and extract active notes
    for time_step, row in enumerate(midi_matrix):
        for note, velocity in enumerate(row):
            if velocity > 0:
                if velocity > 1:
                    time_step_array.append(time_step)
                    velocity_array.append(velocity)

                # Add note_on event
                if velocity > 1:
                    velocity = 1
                track.append(mido.Message('note_on', note=note, velocity=int(velocity * 127), time=time_step))

                # Note off event after one time step (adjust this as needed)
                track.append(mido.Message('note_off', note=note, velocity=0, time=1))

    # Save the MIDI file
    midi.save('output.midi')

    exit()

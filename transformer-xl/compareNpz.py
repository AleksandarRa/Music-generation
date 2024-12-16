import os
import pathlib
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config
import os
import numpy as np
import csv
import pandas as pd

FILENAME = "138"
APPROACH=4
CSVName = f"{APPROACH}approach{FILENAME}NpzData1500.csv"


def append_tuple_to_csv(tuple_data, output_file, columnName, filename_column = False):
    """
    Appends a tuple of arrays as new columns to an existing CSV file, or creates the file if it doesn't exist.

    Parameters:
    - tuple_data: Tuple of arrays with equal lengths.
    - output_file: Path to the output CSV file.
    """
    # Ensure all arrays have the same length
    array_lengths = [len(arr) for arr in tuple_data]
    assert len(set(array_lengths)) == 1, "All arrays in the tuple must have the same length."
    # Convert tuple to a DataFrame
    new_data = pd.DataFrame({columnName[i] : arr for i, arr in enumerate(tuple_data)})

    if filename_column:
        new_data.insert(0, 'Filename', filename_column)
        
    try:
        # Try to read the existing file
        existing_data = pd.read_csv(output_file)
        # Concatenate existing data with new columns
        combined_data = pd.concat([existing_data, new_data], axis=1)
    except FileNotFoundError:
        # If the file doesn't exist, create it with the new data
        combined_data = new_data

    # Write the updated data back to the file
    combined_data.to_csv(output_file, index=False)

def subtract_tuples(tuple1, tuple2):
    return tuple(np.abs(np.array(a, dtype=np.int32) - np.array(b, dtype=np.int32)) for a, b in zip(tuple1, tuple2))

# Initialize MIDI parser and get list of npz files
idx_to_time = get_quant_time()
midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

npz_filenames_fullSong = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/npz/fullSong/").rglob(FILENAME+'.npz'))
file_names_without_extension_fullSong = [file.stem for file in npz_filenames_fullSong]
file_names_array_fullSong = np.array(file_names_without_extension_fullSong)

npz_filenames_1 = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/npz/0-1500/").rglob(FILENAME+'npz_0-1500.npz'))
file_names_without_extension_1 = [file.stem for file in npz_filenames_1]
file_names_array_1 = np.array(file_names_without_extension_1)

npz_filenames_2 = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/npz/1500-3000/").rglob(FILENAME+'npz_1500-3000.npz'))
file_names_without_extension_2 = [file.stem for file in npz_filenames_2]
file_names_array_2 = np.array(file_names_without_extension_2)

sounds_full, deltas_full = zip(*[midi_parser.load_features(filename)
                       for filename in npz_filenames_fullSong])
start=0
end = 1500
sounds_full_1cut = tuple(sound[start:end] for sound in sounds_full)
deltas_full_1cut = tuple(delta[start:end] for delta in deltas_full)

start=1500
end = 3000
sounds_full_2cut = tuple(sound[start:end] for sound in sounds_full)
deltas_full_2cut = tuple(delta[start:end] for delta in deltas_full)

sounds_1, deltas_1 = zip(*[midi_parser.load_features(filename)
                                 for filename in npz_filenames_1])

sounds_2, deltas_2 = zip(*[midi_parser.load_features(filename)
                           for filename in npz_filenames_2])

# Subtracting sounds_full_1cut - sounds_1, and so on
sounds_diff_1 = subtract_tuples(sounds_full_1cut, sounds_1)
deltas_diff_1 = subtract_tuples(deltas_full_1cut, deltas_1)

sounds_diff_2 = subtract_tuples(sounds_full_2cut, sounds_2)
deltas_diff_2 = subtract_tuples(deltas_full_2cut, deltas_2)

columnName =["true label sound 0-1500"]
append_tuple_to_csv(sounds_full_1cut, CSVName, columnName, True)
columnName =["true label delta 0-1500"]
append_tuple_to_csv(deltas_full_1cut, CSVName, columnName)

columnName =["true label sound 1500-3000"]
append_tuple_to_csv(sounds_full_2cut, CSVName , columnName)
columnName =["true label delta 1500-3000"]
append_tuple_to_csv(deltas_full_2cut, CSVName, columnName)

columnName = ["cutted sound 1500-3000"]
append_tuple_to_csv(sounds_1, CSVName, columnName)
columnName = ["cutted delta 0-1500"]
append_tuple_to_csv(deltas_1, CSVName, columnName)

columnName = ["cutted sound 1498-3000"]
append_tuple_to_csv(sounds_2, CSVName, columnName)
columnName = ["cutted delta 1500-3000"]
append_tuple_to_csv(deltas_2, CSVName, columnName)

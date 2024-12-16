import os
import pathlib
from midi_parser import MIDI_parser
from utils import get_quant_time
import config_music as config
import os
import numpy as np
import csv
import pandas as pd


def merge_csv_files(csv_files, output_file):
    """
    Merges multiple CSV files into one and saves the result.

    Parameters:
    - csv_files: List of file paths to the CSV files to be merged.
    - output_file: Path to the output CSV file.
    """
    try:
        # Read all CSV files into dataframes
        dataframes = [pd.read_csv(file) for file in csv_files]

        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Save the merged dataframe to a new CSV file
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

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
        new_data.insert(0, 'Filename', FILENAME)
        
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

APPROACH=4
FILENAMES = ['138', '255', '341', '346']
for FILENAME in FILENAMES:
    CSVName = f"{APPROACH}approach{FILENAME}NpzData1500.csv"

    npz_filenames_fullSong = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/preprocessingTest/1_npz/").rglob(FILENAME+'.npz'))
    file_names_without_extension_fullSong = [file.stem for file in npz_filenames_fullSong]
    file_names_array_fullSong = np.array(file_names_without_extension_fullSong)

    npz_filenames_1 = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/preprocessingTest/3_npz/").rglob(FILENAME+'.npz'))
    file_names_without_extension_1 = [file.stem for file in npz_filenames_1]
    file_names_array_1 = np.array(file_names_without_extension_1)

    npz_filenames_2 = list(pathlib.Path("data/analysedData/"+str(APPROACH)+"approach/preprocessingTest/3_npz/").rglob(FILENAME+'.npz'))
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
    #sounds_diff_1 = subtract_tuples(sounds_full_1cut, sounds_1)
    #deltas_diff_1 = subtract_tuples(deltas_full_1cut, deltas_1)

    #sounds_diff_2 = subtract_tuples(sounds_full_2cut, sounds_2)
    #deltas_diff_2 = subtract_tuples(deltas_full_2cut, deltas_2)

    columnName =["true label sound"]
    append_tuple_to_csv(sounds_full, CSVName, columnName, True)
    columnName =["true label delta"]
    append_tuple_to_csv(deltas_full, CSVName, columnName)

    columnName =["preprocessed sound"]
    append_tuple_to_csv(sounds_1, CSVName , columnName)
    columnName =["preprocessed delta"]
    append_tuple_to_csv(deltas_1, CSVName, columnName)

    #columnName = ["cutted sound 0-1500"]
    #append_tuple_to_csv(sounds_1, CSVName, columnName)
    #columnName = ["cutted delta 0-1500"]
    #append_tuple_to_csv(deltas_1, CSVName, columnName)

    #columnName = ["cutted sound 1500-3000"]
    #append_tuple_to_csv(sounds_2, CSVName, columnName)
    #columnName = ["cutted delta 1500-3000"]
    #append_tuple_to_csv(deltas_2, CSVName, columnName)

    # Example Usage
    csv_files = [f"{APPROACH}approach{FILENAME}NpzData1500.csv", f"{APPROACH}approach{FILENAME}NpzData1500.csv", f"{APPROACH}approach{FILENAME}NpzData1500.csv",f"{APPROACH}approach{FILENAME}NpzData1500.csv"]  # List of CSV file paths
    output_file = f"data/analysedData/{APPROACH}approach/preprocessingTest/3_npz/npzArrayData.csv"  # Path to save the merged CSV

    merge_csv_files(csv_files, output_file)

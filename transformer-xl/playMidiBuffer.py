import os
import subprocess

# Function to play a MIDI file using timidity
def play_midi_file(file_path):
    # Use subprocess to call timidity
    try:
        subprocess.run(['timidity', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while playing MIDI file: {e}")

def play_midi_files_in_directory(directory):
    """Loop over a directory and play each MIDI file in sequence"""
    for i in range(1, 11):
        midi_file = os.path.join(directory, f"{i}_note.midi")
        if os.path.exists(midi_file):
            print(f"Playing {midi_file}...")
            play_midi_file(midi_file)
        else:
            print(f"{midi_file} does not exist.")

# Path to your MIDI file
midi_file_path = 'generated_midis/buffer'

# Play the MIDI file using timidity
play_midi_files_in_directory(midi_file_path)

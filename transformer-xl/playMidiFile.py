import os
import subprocess

# Function to play a MIDI file using timidity
def play_midi_file_with_timidity(file_path):
    # Use subprocess to call timidity
    try:
        subprocess.run(['timidity', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while playing MIDI file: {e}")

# Path to your MIDI file
midi_file_path = 'generated_midis/Bach.mid'

# Play the MIDI file using timidity
play_midi_file_with_timidity(midi_file_path)

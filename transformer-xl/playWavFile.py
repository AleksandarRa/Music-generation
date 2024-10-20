import pygame

# Function to play a MIDI file
def play_midi_file(file_path):
    # Initialize pygame's mixer
    pygame.mixer.init()

    # Load the MIDI file using pygame
    pygame.mixer.music.load(file_path)

    # Play the MIDI file
    pygame.mixer.music.play()

    # Keep playing until the MIDI file ends
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 10ms if it's still playing

# Play the MIDI file
play_midi_file('generated_midis/Bach.wav')

import os
import shutil


def copy_midi_files(source_dir, destination_dir):
    # Ensure destination directory exists, create it if not
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Loop through the source directory to find .mid or .midi files
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                full_file_path = os.path.join(root, file)
                print(f"Copying {file} to {destination_dir}")
                shutil.copy(full_file_path, destination_dir)

    # Remove the source directory once all files are copied
    #shutil.rmtree(source_dir)
    #print(f"Source directory '{source_dir}' deleted.")


source_directory = 'datasets'
destination_directory = 'midi_files'

copy_midi_files(source_directory, destination_directory)

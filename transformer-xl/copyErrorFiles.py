import shutil
import os

# Define paths for the source directory, destination directory, and the list of files
src_dir = "data/npz_seperated/npz_midis/"
dst_dir = "data/npz_seperated_ErrorFiles/npz_midis/"
file_list_path = "nodes/npzErrorFilesMaestro.txt"

# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Read the list of filenames from the file
with open(file_list_path, "r") as file:
    lines = file.readlines()

# Iterate over each line in the file
for line in lines:
    # Extract the filename by splitting on whitespace or '|' and trimming extra whitespace
    npz_file = line.split('|')[0].strip()
    npz_file = npz_file.split(':')[1].strip()

    # Define the full source and destination file paths
    src_file = os.path.join(src_dir, npz_file)
    dst_file = os.path.join(dst_dir, npz_file)

    # Check if the source file exists before copying
    if os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)  # copy2 preserves metadata
        print(f"Copied {npz_file} to {dst_dir}")
    else:
        print(f"{npz_file} does not exist in {src_dir}")

print("File copying complete.")

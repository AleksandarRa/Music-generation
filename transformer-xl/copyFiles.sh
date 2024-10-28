#!/bin/bash

# Source and destination directories
SOURCE_DIR="/home/radovic/Documents/masterarbeit/projects/transformer-xl/data/npz_seperated/npz_smd/"
DEST_USER="aradovic"
DEST_HOST="10.10.41.74"
DEST_DIR="/home2/aradovic/Music-generation/transformer-xl/data/npz_seperated/npz_smd"

# File containing the list of .npz files
FILE_LIST="npzErrorFiles.txt"

# Loop through each line in the file list
while IFS= read -r line; do
    # Extract the file name by splitting on the space and taking the first field
    npz_file=$(echo "$line" | cut -d ' ' -f 3)
    
    # Construct the full source file path
    source_file="${SOURCE_DIR}${npz_file}"
    # Check if the source file exists
    if [[ -f "$source_file" ]]; then
        # Execute the scp command
        scp "$source_file" "${DEST_USER}@${DEST_HOST}:${DEST_DIR}"
        
        # Check the exit status of scp
        if [[ $? -eq 0 ]]; then
            echo "Successfully copied $npz_file to ${DEST_DIR}"
        else
            echo "Failed to copy $npz_file"
        fi
    else
        echo "$npz_file does not exist in $SOURCE_DIR"
    fi
done < "$FILE_LIST"


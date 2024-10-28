#!/bin/bash

#define the script name
SCRIPT="generate_music.py"

# Define the gen_len values
GEN_LENS=(300 1000)

# Define the checkpoint paths
CHECKPOINT_PATHS=(
#	"checkpoints_music/transformerXL/transformerXL_checkpoint20.weights.h5"
#	"checkpoints_music/transformerXL/transformerXL_checkpoint40.weights.h5"
	"checkpoints_music/transformerXL/transformerXL_checkpoint60.weights.h5"
)

# Define the number of songs to generate
N_SONGS=1

# Define the directory with npz files
NPZ_DIR="npz_music"

# Define the destination directory
DST_DIR="generated_midis"

# Loop through each checkpoint path
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
	# Loop through each gen_len value
	for GEN_LEN in "${GEN_LENS[@]}"; do
		echo "Generating music with checkpoint = $CHECKPOINT_PATH and gen_len = $GEN_LEN"
	
		python3 "$SCRIPT" -n "$N_SONGS" \
				  -c "$CHECKPOINT_PATH" \
				  -np "$NPZ_DIR" \
				  -o "$DST_DIR" \
				  -l "$GEN_LEN" \
				  -k 3 \
				  -t 0.35
				  
		echo "Finished generating music with checkpoint = $CHECKPOINT_PATH and gen_len = $GEN_LEN"
		echo "--------------------------------------------------------------------------"
	done
done



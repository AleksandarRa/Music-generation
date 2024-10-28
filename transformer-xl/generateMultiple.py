import subprocess

# Define the script name
script = "generate_music.py"

# Define the gen_len values
gen_lens = [300, 1000, 3000]

# Define the checkpoint paths
checkpoint_paths = [
        "checkpoints_music/transformerXL/transformerXL_checkpoint20.weights.h5",
        "checkpoints_music/transformerXL/transformerXL_checkpoint40.weights.h5"
        "checkpoints_music/transformerXL/transformerXL_checkpoint60.weights.h5"
        ]

# Define other arguments
n_songs = 1
npz_dir = "npz_music"
dst_dir = "generated_midis"
top_k = 3
temperature = 0.35

# Loop through each checkpoint path and gen_len value
for checkpoint_path in checkpoint_paths:
    for gen_len in gen_lens:
        print(f"Generating music with checkpoint = {checkpoint_path} and gen_len = {gen_len}")

        # Construct the command to execute generate_music.py with the specified arguments
        command = [
            "python3", script,
            "-n", str(n_songs),
            "-c", checkpoint_path,
            "-np", npz_dir,
            "-o", dst_dir,
            "-l", str(gen_len),
            "-k", str(top_k),
            "-t", str(temperature)
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print output and error (if any)
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)

        print(f"Finished generating music with checkpoint = {checkpoint_path} and gen_len = {gen_len}")
        print("--------------------------------------------------------------------------")

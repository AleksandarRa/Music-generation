import pandas as pd
# Function to get file size based on user input
def get_file_size():
    filename=""
    # Prompt user for a filename
    while filename != "q.npz":
        filename = input("Enter the filename (e.g., 8717): ").strip()
        filename += ".npz"
        df = pd.read_csv("logs/npz_sizes.csv")
        # Search for the filename in the DataFrame
        matching_row = df[df['Filename'] == filename]

        # Check if a match is found
        if not matching_row.empty:
            filesize = matching_row['Filesize'].values[0]
            print(f"Size: {filesize} ")
        else:
            print(f"File '{filename}' not found in the dataset.")


# Run the function
get_file_size()

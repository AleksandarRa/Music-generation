import glob
import os
import zipfile
import numpy as np
file_directory = "npz_music"
filenames = sorted(glob.glob(os.path.join(file_directory, '*.npz')))
cnt = 1 
for filename in filenames:
    try:
        container = np.load(filename)
        sounds = container['sounds']
        deltas = container['deltas']
    except (OSError, zipfile.BadZipFile, KeyError, NameError) as e:
        print(f'{cnt} : {filename.split('/')[-1]} | {e}')
        try:
            os.remove(filename)
            print(f"Removed invalid file: {filename}")
        except OSError as remove_error:
            print(f"Error removing file {filename}: {remove_error}")
        cnt +=1
        continue


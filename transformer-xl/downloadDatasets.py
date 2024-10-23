from midi_parser import MIDI_parser
import config_music as config
from utils import get_quant_time
import numpy as np
import argparse
import pathlib
import dload

MIDI_DIR = "datasets/"

if __name__ == '__main__':
    if not pathlib.Path(MIDI_DIR).exists():
        pathlib.Path(MIDI_DIR).mkdir(parents=True, exist_ok=True)
    else:
        assert pathlib.Path(MIDI_DIR).is_dir()

    datasets = [
        #{"name": "maestro",
         #"url": "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"},
        {"name": "SMD", "url": "https://zenodo.org/record/13753319/files/SMD-piano_v2.zip?download=1"}
    ]

    for dataset in datasets:
        print('Downloading dataset ' + dataset['name'])
        dataset_path = MIDI_DIR + dataset['name']
        if not pathlib.Path(dataset_path).exists():
            pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        else:
            assert pathlib.Path(dataset_path).is_dir()
        dload.save_unzip(dataset['url'], dataset_path)

    print('Download finished')
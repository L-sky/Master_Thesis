import os
import argparse
import numpy as np
import pymatgen
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")

args = parser.parse_args()
mp_dir = args.mp_dir

mp_save_dir = mp_dir

index = np.load(os.path.join(mp_dir, 'index.npy'))
run_type = np.load(os.path.join(mp_dir, 'meta', 'run_type.npy'))

valid_run_type = np.logical_or(run_type == 'GGA', run_type == 'GGA+U')
index_subset = index[valid_run_type]

np.save(os.path.join(mp_save_dir, f"index_subset.npy"), index_subset)


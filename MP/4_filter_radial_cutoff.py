import os
import argparse
import numpy as np
import pymatgen
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")
parser.add_argument("radial_cutoff", type=float, help="Radius of sphere that decides neighborhood")

args = parser.parse_args()
mp_dir = args.mp_dir
r_cut = args.radial_cutoff

mp_save_dir = os.path.join(mp_dir, "meta_derived")

index = np.load(os.path.join(mp_dir, 'index_subset.npy'))

nnd = np.load(os.path.join(mp_dir, 'meta_derived', 'closest_neighbor_distances.npy'), allow_pickle=True)
nnd_max = np.array([max(nnd_list) for nnd_list in nnd])

valid_r_cut = nnd_max < r_cut 
index_subset_r_cut = index[valid_r_cut]

np.save(os.path.join(mp_save_dir, f"index_presubset_{r_cut}.npy"), index_subset_r_cut)


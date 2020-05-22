import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")
parser.add_argument("radial_cutoff", type=float, help="Radius of sphere that decides neighborhood")

args = parser.parse_args()
mp_dir = args.mp_dir
r_cut = args.radial_cutoff

mp_dir_meta = os.path.join(mp_dir, 'meta_derived')
mp_dir_save = os.path.join(mp_dir, 'meta_derived')

index = np.load(os.path.join(mp_dir_meta, f'index_presubset_{r_cut}.npy'))
diameters = np.load(os.path.join(mp_dir_meta, f'diameters_{r_cut}.npy'))

index = index[diameters != -1]
diameters = diameters[diameters != -1]

np.save(os.path.join(mp_dir_save, f'index_connected_{r_cut}.npy'), index)
np.save(os.path.join(mp_dir_save, f'diameters_connected_{r_cut}.npy'), diameters)


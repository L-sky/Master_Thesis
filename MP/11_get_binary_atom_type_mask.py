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

index = np.load(os.path.join(mp_dir, 'meta_derived', f'index_connected_{r_cut}.npy'))
mp_cif_dir = os.path.join(mp_dir, "cif")
mp_save_dir = os.path.join(mp_dir, f"derived_radial_cutoff_{r_cut}")

def get_max_atomic_number(cif_paths):
    max_atomic_number = -1
    for cif_path in tqdm(cif_paths):
        structure = pymatgen.Structure.from_file(cif_path)
        max_atomic_number = max(max_atomic_number, max(structure.atomic_numbers))
    return max_atomic_number     

def process_cif(cif_path):
    structure = pymatgen.Structure.from_file(cif_path)
    return np.array(structure.atomic_numbers)


cif_paths = [os.path.join(mp_cif_dir, filename) for filename in index]
max_atomic_number = get_max_atomic_number(cif_paths)
atom_type_mask = np.zeros((len(cif_paths), max_atomic_number+1), dtype=np.bool)

for i, cif_path in enumerate(tqdm(cif_paths)):
    atom_type_mask[i, process_cif(cif_path)] = True
    
np.save(os.path.join(mp_save_dir, "atom_type_mask.npy"), atom_type_mask)

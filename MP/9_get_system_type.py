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

def process_cif(cif_path):
    structure = pymatgen.Structure.from_file(cif_path)
    analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(structure)
    spacegroup_symbol, international_number = structure.get_space_group_info()
    return analyzer.get_crystal_system(), analyzer.get_lattice_type(), spacegroup_symbol, international_number 


crystal_system_list = []
lattice_type_list = []
spacegroup_symbol_list = []
international_number_list = []

cif_paths = [os.path.join(mp_cif_dir, filename) for filename in index]
for cif_path in tqdm(cif_paths):
    crystal_system, lattice_type, spacegroup_symbol, international_number = process_cif(cif_path)
    crystal_system_list.append(crystal_system)
    lattice_type_list.append(lattice_type)
    spacegroup_symbol_list.append(spacegroup_symbol)
    international_number_list.append(international_number)

np.save(os.path.join(mp_save_dir, "crystal_system.npy"), np.array(crystal_system_list))
np.save(os.path.join(mp_save_dir, "lattice_type.npy"), np.array(lattice_type_list))
np.save(os.path.join(mp_save_dir, "spacegroup_symbol.npy"), np.array(spacegroup_symbol_list))
np.save(os.path.join(mp_save_dir, "international_number.npy"), np.array(international_number_list))

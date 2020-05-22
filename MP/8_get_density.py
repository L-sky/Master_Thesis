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

if not os.path.isdir(mp_save_dir):
    os.mkdir(mp_save_dir)

def process_cif(cif_path):
    structure = pymatgen.Structure.from_file(cif_path)
    volume = structure.volume
    n_atoms = structure.num_sites
    density = structure.density 				# g/cm^3
    spatial_density = n_atoms / volume 				# atoms/A^3
    return volume, n_atoms, density, spatial_density


n_atoms_list = []
unit_cell_volume_list = []
density_list = []
spatial_density_list = []

cif_paths = [os.path.join(mp_cif_dir, filename) for filename in index]
for cif_path in tqdm(cif_paths):
    unit_cell_volume, n_atoms, density, spatial_density = process_cif(cif_path)

    n_atoms_list.append(n_atoms)
    unit_cell_volume_list.append(unit_cell_volume)
    density_list.append(density)
    spatial_density_list.append(spatial_density)

np.save(os.path.join(mp_save_dir, "n_atoms.npy"), np.array(n_atoms_list))
np.save(os.path.join(mp_save_dir, "volumes.npy"), np.array(unit_cell_volume_list))
np.save(os.path.join(mp_save_dir, "densities.npy"), np.array(density_list))
np.save(os.path.join(mp_save_dir, "spatial_densities.npy"), np.array(spatial_density_list))

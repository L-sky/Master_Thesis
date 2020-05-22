import os
import pickle
import argparse
import pathlib

from os import mkdir
from os.path import join, isdir
from shutil import rmtree

import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mp_pkl_dir", help="directory with pkl file containing dataset")
parser.add_argument("mp_dir", help="directory to store dataset extracted from pkl file")

args = parser.parse_args()
mp_pkl_dir = args.mp_pkl_dir
mp_dir = args.mp_dir 

with open(os.path.join(mp_pkl_dir, "MP_data.pkl"), 'rb') as fp:
    data = pickle.load(fp)

mp_cif_dir = join(mp_dir, 'cif')
mp_meta_dir = join(mp_dir, 'meta')
mp_base_properties_dir = join(mp_dir, 'base_properties')
mp_additional_properties_dir = join(mp_dir, 'additional_properties')

mp_elasticity_dir = join(mp_additional_properties_dir, 'elasticity')
mp_diel_dir = join(mp_additional_properties_dir, 'diel')
mp_piezo_dir = join(mp_additional_properties_dir, 'piezo')

if isdir(mp_dir):
    rmtree(mp_dir)      # clean-up after supposedly failed prior extraction

pathlib.Path(mp_dir).mkdir(parents=True, exist_ok=True)
mkdir(mp_cif_dir)
mkdir(mp_meta_dir)
mkdir(mp_base_properties_dir)
mkdir(mp_additional_properties_dir)
mkdir(mp_elasticity_dir)
mkdir(mp_diel_dir)
mkdir(mp_piezo_dir)

task_id_list = []
e_above_hull_list = []
run_type_list = []
oxide_type_list = []
functional_list = []
pot_type_list = []
pretty_formula_list = []
unit_cell_formula_list = []
band_gap_list = []
formation_energy_per_atom_list = []

is_hubbard_list = []
hubbards_list = []
is_compatible_list = []
theoretical_list = []

elasticity_id_list = []
piezo_id_list = []
diel_id_list = []

elasticity_G_Reuss_list = []
elasticity_G_VRH_list = []
elasticity_G_Voigt_list = []
elasticity_G_Voigt_Reuss_Hill_list = []
elasticity_K_Reuss_list = []
elasticity_K_VRH_list = []
elasticity_K_Voigt_list = []
elasticity_K_Voigt_Reuss_Hill_list = []
elasticity_elastic_anisotropy_list = []
elasticity_elastic_tensor_list = []
elasticity_homogeneous_poisson_list = []
elasticity_poisson_ratio_list = []
elasticity_universal_anisotropy_list = []
elasticity_elastic_tensor_original_list = []
elasticity_compliance_tensor_list = []
elasticity_warnings_list = []
elasticity_nsites_list = []

diel_e_electronic_list = []
diel_e_total_list = []
diel_n_list = []
diel_poly_electronic_list = []
diel_poly_total_list = []

piezo_eij_max_list = []
piezo_piezoelectric_tensor_list = []
piezo_v_max_list = []

index = []

elasticity_id = 0
piezo_id = 0
diel_id = 0
for entry in tqdm(data):
    task_id = entry['task_id']
    e_above_hull = entry['e_above_hull']
    run_type = entry['run_type']
    oxide_type = entry['oxide_type']
    functional = entry['pseudo_potential']['functional']
    pot_type = entry['pseudo_potential']['pot_type']
    pretty_formula = entry['pretty_formula']
    unit_cell_formula = entry['unit_cell_formula']
    band_gap = entry['band_gap']
    formation_energy_per_atom = entry['formation_energy_per_atom']
    elasticity = entry['elasticity']
    piezo = entry['piezo']
    diel = entry['diel']
    is_hubbard = entry['is_hubbard']
    hubbards = entry['hubbards']
    is_compatible = entry['is_compatible']
    theoretical = entry['theoretical']
    cif = entry['cif']

    task_id_list.append(task_id)
    e_above_hull_list.append(e_above_hull)
    run_type_list.append(run_type)
    oxide_type_list.append(oxide_type)
    functional_list.append(functional)
    pot_type_list.append(pot_type)
    pretty_formula_list.append(pretty_formula)
    unit_cell_formula_list.append(unit_cell_formula)
    band_gap_list.append(band_gap)
    formation_energy_per_atom_list.append(formation_energy_per_atom)

    is_hubbard_list.append(is_hubbard)
    hubbards_list.append(hubbards)
    is_compatible_list.append(is_compatible)
    theoretical_list.append(theoretical)

    if elasticity is None:
        elasticity_id_list.append(-1)       # aka point to Nothing
    else:
        elasticity_id_list.append(elasticity_id)
        elasticity_id = elasticity_id + 1
        elasticity_G_Reuss_list.append(elasticity['G_Reuss'])
        elasticity_G_VRH_list.append(elasticity['G_VRH'])
        elasticity_G_Voigt_list.append(elasticity['G_Voigt'])
        elasticity_G_Voigt_Reuss_Hill_list.append(elasticity['G_Voigt_Reuss_Hill'])
        elasticity_K_Reuss_list.append(elasticity['K_Reuss'])
        elasticity_K_VRH_list.append(elasticity['K_VRH'])
        elasticity_K_Voigt_list.append(elasticity['K_Voigt'])
        elasticity_K_Voigt_Reuss_Hill_list.append(elasticity['K_Voigt_Reuss_Hill'])
        elasticity_elastic_anisotropy_list.append(elasticity['elastic_anisotropy'])
        elasticity_elastic_tensor_list.append(elasticity['elastic_tensor'])
        elasticity_homogeneous_poisson_list.append(elasticity['homogeneous_poisson'])
        elasticity_poisson_ratio_list.append(elasticity['poisson_ratio'])
        elasticity_universal_anisotropy_list.append(elasticity['universal_anisotropy'])
        elasticity_elastic_tensor_original_list.append(elasticity['elastic_tensor_original'])
        elasticity_compliance_tensor_list.append(elasticity['compliance_tensor'])
        elasticity_warnings_list.append(elasticity['warnings'])
        elasticity_nsites_list.append(elasticity['nsites'])

    if diel is None:
        diel_id_list.append(-1)
    else:
        diel_id_list.append(diel_id)
        diel_id = diel_id + 1
        diel_e_electronic_list.append(diel['e_electronic'])
        diel_e_total_list.append(diel['e_total'])
        diel_n_list.append(diel['n'])
        diel_poly_electronic_list.append(diel['poly_electronic'])
        diel_poly_total_list.append(diel['poly_total'])

    if piezo is None:
        piezo_id_list.append(-1)
    else:
        piezo_id_list.append(piezo_id)
        piezo_id = piezo_id + 1
        piezo_eij_max_list.append(piezo['eij_max'])
        piezo_piezoelectric_tensor_list.append(piezo['piezoelectric_tensor'])
        piezo_v_max_list.append(piezo['v_max'])

    index.append(f'{task_id}.cif')
    with open(join(mp_cif_dir, f'{task_id}.cif'), 'w') as cif_file:
        cif_file.write(cif)

np.save(join(mp_dir, 'index.npy'), np.array(index))
np.save(join(mp_dir, 'task_id.npy'), np.array(task_id_list))
np.save(join(mp_dir, 'names.npy'), np.array(pretty_formula_list))                   # duplicate of pretty formula under different alias is intentional

# metadata
np.save(join(mp_meta_dir, 'e_above_hull.npy'), np.array(e_above_hull_list))
np.save(join(mp_meta_dir, 'run_type.npy'), np.array(run_type_list))
np.save(join(mp_meta_dir, 'oxide_type.npy'), np.array(oxide_type_list))
np.save(join(mp_meta_dir, 'functional.npy'), np.array(functional_list))
np.save(join(mp_meta_dir, 'pot_type.npy'), np.array(pot_type_list))
np.save(join(mp_meta_dir, 'pretty_formula.npy'), np.array(pretty_formula_list))
np.save(join(mp_meta_dir, 'unit_cell_formula.npy'), np.array(unit_cell_formula_list))
np.save(join(mp_meta_dir, 'is_hubbard.npy'), np.array(is_hubbard_list))
np.save(join(mp_meta_dir, 'hubbards.npy'), hubbards_list)                           # list of dicts
np.save(join(mp_meta_dir, 'is_compatible.npy'), np.array(is_compatible_list))
np.save(join(mp_meta_dir, 'theoretical.npy'), np.array(theoretical_list))

# base properties (known for all structures)
torch.save(torch.tensor(band_gap_list, dtype=torch.float64), join(mp_base_properties_dir, 'band_gap.pth'))
torch.save(torch.tensor(formation_energy_per_atom_list, dtype=torch.float64), join(mp_base_properties_dir, 'formation_energy_per_atom.pth'))

# additional properties (known only for some structures)

# indexing
torch.save(torch.tensor(elasticity_id_list, dtype=torch.int64), join(mp_additional_properties_dir, 'elasticity_id.pth'))
torch.save(torch.tensor(diel_id_list, dtype=torch.int64), join(mp_additional_properties_dir, 'diel_id.pth'))
torch.save(torch.tensor(piezo_id_list, dtype=torch.int64), join(mp_additional_properties_dir, 'piezo_id.pth'))

# values: elasticity
torch.save(torch.tensor(elasticity_G_Reuss_list, dtype=torch.float64), join(mp_elasticity_dir, 'G_Reuss.pth'))
torch.save(torch.tensor(elasticity_G_VRH_list, dtype=torch.float64), join(mp_elasticity_dir, 'G_VRH.pth'))
torch.save(torch.tensor(elasticity_G_Voigt_list, dtype=torch.float64), join(mp_elasticity_dir, 'G_Voigt.pth'))
torch.save(torch.tensor(elasticity_G_Voigt_Reuss_Hill_list, dtype=torch.float64), join(mp_elasticity_dir, 'G_Voigt_Reuss_Hill.pth'))
torch.save(torch.tensor(elasticity_K_Reuss_list, dtype=torch.float64), join(mp_elasticity_dir, 'K_Reuss.pth'))
torch.save(torch.tensor(elasticity_K_VRH_list, dtype=torch.float64), join(mp_elasticity_dir, 'K_VRH.pth'))
torch.save(torch.tensor(elasticity_K_Voigt_list, dtype=torch.float64), join(mp_elasticity_dir, 'K_Voigt.pth'))
torch.save(torch.tensor(elasticity_K_Voigt_Reuss_Hill_list, dtype=torch.float64), join(mp_elasticity_dir, 'K_Voigt_Reuss_Hill.pth'))
torch.save(torch.tensor(elasticity_elastic_anisotropy_list, dtype=torch.float64), join(mp_elasticity_dir, 'elastic_anisotropy.pth'))
torch.save(torch.tensor(elasticity_elastic_tensor_list, dtype=torch.float64), join(mp_elasticity_dir, 'elastic_tensor.pth'))
torch.save(torch.tensor(elasticity_homogeneous_poisson_list, dtype=torch.float64), join(mp_elasticity_dir, 'homogeneous_poisson.pth'))
torch.save(torch.tensor(elasticity_poisson_ratio_list, dtype=torch.float64), join(mp_elasticity_dir, 'poisson_ratio.pth'))
torch.save(torch.tensor(elasticity_universal_anisotropy_list, dtype=torch.float64), join(mp_elasticity_dir, 'universal_anisotropy.pth'))
torch.save(torch.tensor(elasticity_elastic_tensor_original_list, dtype=torch.float64), join(mp_elasticity_dir, 'elastic_tensor_original.pth'))
torch.save(torch.tensor(elasticity_compliance_tensor_list, dtype=torch.float64), join(mp_elasticity_dir, 'compliance_tensor.pth'))
np.save(join(mp_elasticity_dir, 'warnings.npy'), elasticity_warnings_list)
torch.save(torch.tensor(elasticity_nsites_list, dtype=torch.int64), join(mp_elasticity_dir, 'nsites.pth'))

# values: diel
torch.save(torch.tensor(diel_e_electronic_list, dtype=torch.float64), join(mp_diel_dir, 'e_electronic.pth'))
torch.save(torch.tensor(diel_e_total_list, dtype=torch.float64), join(mp_diel_dir, 'e_total.pth'))
torch.save(torch.tensor(diel_n_list, dtype=torch.float64), join(mp_diel_dir, 'n.pth'))
torch.save(torch.tensor(diel_poly_electronic_list, dtype=torch.float64), join(mp_diel_dir, 'poly_electronic.pth'))
torch.save(torch.tensor(diel_poly_total_list, dtype=torch.float64), join(mp_diel_dir, 'poly_total.pth'))

# values: piezo
torch.save(torch.tensor(piezo_eij_max_list, dtype=torch.float64), join(mp_piezo_dir, 'eij_max.pth'))
torch.save(torch.tensor(piezo_piezoelectric_tensor_list, dtype=torch.float64), join(mp_piezo_dir, 'piezoelectric_tensor.pth'))
torch.save(torch.tensor(piezo_v_max_list, dtype=torch.float64), join(mp_piezo_dir, 'v_max.pth'))

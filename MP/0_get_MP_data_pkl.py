import os
import pickle
import argparse

from pymatgen import MPRester

parser = argparse.ArgumentParser()
parser.add_argument("api_key", help="Materials Project API KEY")
parser.add_argument("save_path", help="directory to save pkl file with dataset")

args = parser.parse_args()
api_key = args.api_key
save_path = args.save_path 

mpr = MPRester(api_key)
MP_data = mpr.query(criteria={"cif": {"$exists": True}},
                    properties=["task_id", "pretty_formula", "unit_cell_formula",
				"e_above_hull", "run_type", "pseudo_potential", "oxide_type",
				"band_gap", "formation_energy_per_atom",
				"elasticity", "piezo", "diel",
				"is_hubbard", "hubbards", "is_compatible",
				"theoretical",
				"cif"])
    
with open(os.path.join(save_path, "MP_data.pkl"), 'wb') as fp:
    pickle.dump(MP_data, fp)

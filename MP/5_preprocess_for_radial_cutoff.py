import os
import argparse

from e3_layer.util.dataset.crystals import CrystalCIF

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")
parser.add_argument("radial_cutoff", type=float, help="Radius of sphere that decides neighborhood")

args = parser.parse_args()
mp_dir = args.mp_dir
r_cut = args.radial_cutoff

index_rel_path = os.path.join('meta_derived', f'index_presubset_{r_cut}.npy')

CrystalCIF.preprocess(mp_dir, index_rel_path, r_cut)



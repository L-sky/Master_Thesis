import torch
import numpy as np
import igraph

import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")
parser.add_argument("radial_cutoff", type=float, help="Radius of sphere that decides neighborhood")

args = parser.parse_args()
mp_dir = args.mp_dir
r_cut = args.radial_cutoff

mp_prep_dir = os.path.join(mp_dir, "preprocessed", f"radial_cutoff_{r_cut}") 
mp_save_dir = os.path.join(mp_dir, "meta_derived")

def create_graph(a_list, b_list):
    edge_list = list(zip(a_list, b_list))
    edge_list = [edge for edge in edge_list if edge[0] != edge[1]] # filter out self loops
    g = igraph.Graph(edge_list)
    return g

partitions = torch.load(os.path.join(mp_prep_dir, 'ab_p_partitions.pth'))
map_ab_p_to_a = torch.load(os.path.join(mp_prep_dir, 'map_ab_p_to_a.pth'))
map_ab_p_to_b = torch.load(os.path.join(mp_prep_dir, 'map_ab_p_to_b.pth'))

diameters = []

for (start, end) in tqdm(partitions):
    a_list = map_ab_p_to_a[start:end].numpy().tolist()
    b_list = map_ab_p_to_b[start:end].numpy().tolist()
    n_vertices = max(a_list) + 1
    g = create_graph(a_list, b_list)
    diameter = g.get_diameter(directed=False, unconn=False)
    diameter = -1 if isinstance(diameter, int) else len(diameter) - 1  # not number of vertices in path, but edges
    diameters.append(diameter)

np.save(os.path.join(mp_save_dir, f"diameters_{r_cut}.npy"), np.array(diameters))

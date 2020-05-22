import os
import argparse
import numpy as np
import pymatgen
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mp_dir", help="Root directory with Materials Project dataset")

args = parser.parse_args()
mp_dir = args.mp_dir

index = np.load(os.path.join(mp_dir, 'index_subset.npy'))
mp_cif_dir = os.path.join(mp_dir, "cif")
mp_save_dir = os.path.join(mp_dir, "meta_derived")

if not os.path.isdir(mp_save_dir):
    os.mkdir(mp_save_dir)

inner_radii = []
outer_radii = []
closest_neighbor_distances = []
closest_neighbor_distances_2 = []
closest_neighbor_distances_3 = []
closest_neighbor_distances_4 = []
closest_neighbor_distances_5 = []
closest_neighbor_distances_6 = []

def process_cif(cif_path):
    triplet = np.array([1, 0, -1]).reshape(3,1)
    
    structure = pymatgen.Structure.from_file(cif_path)
    A, B, C = structure.lattice.matrix
    
    # radii over unit cell
    AxB = np.cross(A, B)
    AxC = np.cross(A, C)
    BxC = np.cross(B, C)
    Ap = np.abs(np.dot(BxC, A))/np.linalg.norm(BxC)
    Bp = np.abs(np.dot(AxC, B))/np.linalg.norm(AxC)    
    Cp = np.abs(np.dot(AxB, C))/np.linalg.norm(AxB)

    r = np.min([Ap, Bp, Cp])
    R = np.max([np.linalg.norm(A+B+C), np.linalg.norm(A+B-C), np.linalg.norm(A-B+C), np.linalg.norm(-A+B+C)])

    # closest neighbor distances 
    At = A.reshape(1,3) * triplet
    Bt = B.reshape(1,3) * triplet
    Ct = C.reshape(1,3) * triplet
    Dt = At.reshape(3, 1, 1, 3) + Bt.reshape(1,3,1,3) + Ct.reshape(1,1,3,3)
    box = Dt.reshape(27, 3)

    positions = np.array([site.coords for site in structure.sites])					# [a, 3]
    diff_positions = positions.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)				# [a, a, 3]
    diff_positions_box = diff_positions.reshape(1, *diff_positions.shape) + box.reshape(27, 1, 1, 3)	# [27, a, a, 3]
    abs_diff_positions_box = np.linalg.norm(diff_positions_box, axis=-1)				# [27, a, a]
    
    abs_diff_positions_box[13] += 1e7 * np.eye(abs_diff_positions_box.shape[1])				# exclude self-loops
    abs_diff_positions_box = abs_diff_positions_box.reshape(-1, abs_diff_positions_box.shape[2])	# [27 * a, a]
    abs_diff_positions_box_part_sorted = np.sort(abs_diff_positions_box, axis=0)			# [27 * a, a]
    # closest_neighbor_distances_subset = np.min(abs_diff_positions_box, axis=0)
    # closest_neighbor_distances.append(closest_neighbor_distances_subset)

    return r, R, abs_diff_positions_box_part_sorted[:6]							# [6, a]


cif_paths = [os.path.join(mp_cif_dir, filename) for filename in index]
for cif_path in tqdm(cif_paths):
    r, R, abs_diff_positions_box_part_sorted = process_cif(cif_path)

    inner_radii.append(r)
    outer_radii.append(R)
    
    closest_neighbor_distances.append(abs_diff_positions_box_part_sorted[0].tolist())
    closest_neighbor_distances_2.append(abs_diff_positions_box_part_sorted[1].tolist())
    closest_neighbor_distances_3.append(abs_diff_positions_box_part_sorted[2].tolist())
    closest_neighbor_distances_4.append(abs_diff_positions_box_part_sorted[3].tolist())
    closest_neighbor_distances_5.append(abs_diff_positions_box_part_sorted[4].tolist())
    closest_neighbor_distances_6.append(abs_diff_positions_box_part_sorted[5].tolist())


np.save(os.path.join(mp_save_dir, "inner_radii.npy"), np.array(inner_radii))
np.save(os.path.join(mp_save_dir, "outer_radii.npy"), np.array(outer_radii))
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances.npy"), closest_neighbor_distances)
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances_2.npy"), closest_neighbor_distances_2)
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances_3.npy"), closest_neighbor_distances_3)
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances_4.npy"), closest_neighbor_distances_4)
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances_5.npy"), closest_neighbor_distances_5)
np.save(os.path.join(mp_save_dir, "closest_neighbor_distances_6.npy"), closest_neighbor_distances_6)

from os.path import join, isfile, isdir
from os import mkdir
from shutil import rmtree

import torch
from torch.utils.data import Dataset

import numpy as np
import pymatgen

from tqdm import tqdm



class CrystalCIF(Dataset):
    """
    root
    |_______index.npy
    |_______names.npy
    |_______property0.pth
    |_______property1.pth
    |_______cif
    |        |____name0.cif
    |        |____name1.cif
    |
    |_______preprocessed (script creates this folder and its contents)
             |____geometries.pth
             |____atomic_charges.pth
             |____lattice_params.pth
             |____a_partitions.pth
             |____radial_cutoff_{radial_cutoff_value}
                    |_____radii.pth
                    |_____map_ab_p_to_a.pth
                    |_____map_ab_p_to_b.pth
                    |_____partitions.pth
    """
    def __init__(self, root, radial_cutoff, material_properties=None):
        """
        :param root: string, path to the root directory of dataset
        :param radial_cutoff: float, radius of sphere
        :param material_properties: (optional) list of file paths containing additional properties, one type of property per file
        """
        preprocessed_dir = join(root, 'preprocessed')
        preprocessed_radius_dir = join(preprocessed_dir, f'radial_cutoff_{radial_cutoff}')

        if (
                isdir(preprocessed_radius_dir)
                and (not isfile(join(preprocessed_radius_dir, 'radii.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'map_ab_p_to_a.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'map_ab_p_to_b.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'n_norm.pth'))
                     or not isfile(join(preprocessed_radius_dir, 'ab_p_partitions.pth')))
        ):
            rmtree(preprocessed_radius_dir)
            CrystalCIF.preprocess(root, radial_cutoff)
        elif not isdir(preprocessed_radius_dir):
            CrystalCIF.preprocess(root, radial_cutoff)
        else:
            pass

        self.names = np.load(join(root, 'names.npy'))
        self.size = len(self.names)
        self.geometries = torch.load(join(preprocessed_dir, 'geometries.pth'))
        self.atomic_charges = torch.load(join(preprocessed_dir, 'atomic_charges.pth'))
        self.lattice_params = torch.load(join(preprocessed_dir, 'lattice_params.pth'))
        self.a_partitions = torch.load(join(preprocessed_dir, 'a_partitions.pth'))

        self.radii = torch.load(join(preprocessed_radius_dir, 'radii.pth'))
        self.map_ab_p_to_a = torch.load(join(preprocessed_radius_dir, 'map_ab_p_to_a.pth'))
        self.map_ab_p_to_b = torch.load(join(preprocessed_radius_dir, 'map_ab_p_to_b.pth'))
        self.n_norm = torch.load(join(preprocessed_radius_dir, 'n_norm.pth'))
        self.ab_p_partitions = torch.load(join(preprocessed_radius_dir, 'ab_p_partitions.pth'))

        if material_properties:
            self.properties = torch.stack([torch.load(join(root, property_path)) for property_path in material_properties], dim=1)
        else:
            self.properties = None

    def __getitem__(self, item_id):
        properties = None if self.properties is None else self.properties[item_id]
        ab_p_start, ab_p_end = self.ab_p_partitions[item_id]
        a_start, a_end = self.a_partitions[item_id]
        return self.names[item_id], self.radii[ab_p_start:ab_p_end], self.map_ab_p_to_a[ab_p_start:ab_p_end], self.map_ab_p_to_b[ab_p_start:ab_p_end],\
               self.n_norm[a_start:a_end], self.geometries[item_id], self.atomic_charges[a_start:a_end], self.lattice_params[item_id], properties

    def __len__(self):
        return self.size

    @staticmethod
    def preprocess(root, index_file_name='index.npy', radial_cutoff=None):
        """
        Allows calls without class instance: CrystalCIF.preprocess(...).
        :param root: string, path to the root directory of dataset
        :param radial_cutoff: float, (optional) radius of sphere
        """
        # region 0. Set up
        preprocessed_dir = join(root, 'preprocessed')
        if not isdir(preprocessed_dir):
            mkdir(preprocessed_dir)

        if radial_cutoff:
            radial_cutoff_dir = join(preprocessed_dir, f'radial_cutoff_{radial_cutoff}')
            if not isdir(radial_cutoff_dir):
                mkdir(radial_cutoff_dir)
        # endregion

        # region 1. Init
        index = np.load(join(root, index_file_name))

        site_a_coords_list = []
        atomic_charges_list = []
        lattice_params_list = []

        if radial_cutoff:
            map_ab_p_to_a_list = []
            map_ab_p_to_b_list = []
            radii_list = []
            n_norm_list = []
            ab_p_partitions_list = []
            a_partitions_list = []
        # endregion

        # region 2. Process
        ab_p_partition_start = 0
        a_partition_start = 0
        for file_rel_path in tqdm(index, desc=f"Preprocessing for {radial_cutoff}"):
            structure = pymatgen.Structure.from_file((join(root, 'cif', file_rel_path)))

            site_a_coords_entry = torch.stack([torch.from_numpy(site.coords) for site in structure.sites])

            site_a_coords_list.append(site_a_coords_entry)
            atomic_charges_list.extend([atom.number for atom in structure.species])
            lattice_params_list.append(structure.lattice.abc + structure.lattice.angles)

            if radial_cutoff:
                radii_proxy_list = []
                for a, site_a_coords in enumerate(site_a_coords_entry):
                    nei = structure.get_sites_in_sphere(site_a_coords.numpy(), radial_cutoff, include_index=True)
                    assert nei, f"Encountered empty nei for {file_rel_path}: {site_a_coords}"

                    entry_data = [entry[2] for entry in nei]
                    map_ab_p_to_a_list.extend([a]*len(entry_data))
                    map_ab_p_to_b_list.extend(entry_data)

                    site_b_coords = np.array([entry[0].coords for entry in nei])                    # [r_part_a, 3]
                    site_a_coords = np.array(site_a_coords).reshape(1, 3)                           # [1, 3]
                    radii_proxy_list.append(site_b_coords - site_a_coords)                          # implicit broadcasting of site_a_coords
                    n_norm_list.append(1.0 / np.sqrt(len(nei)))

                radii_proxy = np.concatenate(radii_proxy_list)                                      # [r, 3]
                radii_proxy[np.linalg.norm(radii_proxy, ord=2, axis=-1) < 1e-10] = 0.
                radii_list.append(radii_proxy)

                ab_p_partitions_list.append((ab_p_partition_start, ab_p_partition_start + radii_proxy.shape[0]))
                ab_p_partition_start += radii_proxy.shape[0]

                a_partitions_list.append((a_partition_start, a_partition_start + site_a_coords_entry.shape[0]))
                a_partition_start += site_a_coords_entry.shape[0]
        # endregion

        # region 3. Post-process
        lattice_params = torch.tensor(lattice_params_list)
        del lattice_params_list

        if radial_cutoff:
            map_ab_p_to_a = torch.tensor(map_ab_p_to_a_list, dtype=torch.int32)
            del map_ab_p_to_a_list

            map_ab_p_to_b = torch.tensor(map_ab_p_to_b_list, dtype=torch.int32)
            del map_ab_p_to_b_list

            radii = torch.from_numpy(np.concatenate(radii_list))
            del radii_list

            n_norm = torch.tensor(n_norm_list, dtype=torch.float64)
            del n_norm_list

            atomic_charges = torch.tensor(atomic_charges_list, dtype=torch.int32)
            del atomic_charges_list

            ab_p_partitions = torch.tensor(ab_p_partitions_list, dtype=torch.long)
            del ab_p_partitions_list

            a_partitions = torch.tensor(a_partitions_list, dtype=torch.long)
            del a_partitions_list
        # endregion

        # region 4. Store
        torch.save(site_a_coords_list, join(preprocessed_dir, 'geometries.pth'))                    # list of z tensors [a_i, 3]            - xyz
        torch.save(atomic_charges, join(preprocessed_dir, 'atomic_charges.pth'))
        torch.save(lattice_params, join(preprocessed_dir, 'lattice_params.pth'))                    # [z, 6]                                - xyz and angles (in degrees)
        torch.save(a_partitions, join(preprocessed_dir, 'a_partitions.pth'))                        # tensor [n_structures, 2]              - start/end of a_slice
        del site_a_coords_list, atomic_charges, lattice_params, a_partitions

        if radial_cutoff:
            torch.save(map_ab_p_to_a, join(radial_cutoff_dir, 'map_ab_p_to_a.pth'))                    # tensor [sum(r_i)]
            del map_ab_p_to_a

            torch.save(map_ab_p_to_b, join(radial_cutoff_dir, 'map_ab_p_to_b.pth'))                    # tensor [sum(r_i)]
            del map_ab_p_to_b

            torch.save(radii, join(radial_cutoff_dir, 'radii.pth'))                                    # tensor [sum(r_i), 3]                  - xyz
            del radii

            torch.save(n_norm, join(radial_cutoff_dir, 'n_norm.pth'))
            del n_norm

            torch.save(ab_p_partitions, join(radial_cutoff_dir, 'ab_p_partitions.pth'))                # tensor [z, 2]                         - start/end of ab_p slices
            del ab_p_partitions
        # endregion

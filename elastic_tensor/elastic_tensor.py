import torch
import numpy as np


class ElasticTensor():
    def __init__(self, voigt):
        self.voigt = voigt
        self.voigt_map = {(1,1): 1, (2,2): 2, (3,3): 3, (2,3): 4, (1,3): 5, (1,2): 6}
        self.cov_mat = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
        self.cov_factor = -0.5*np.sqrt(3/np.pi)

    def voigt_to_cartesian(self):
        voigt = self.voigt.clone()
        voigt_map = self.voigt_map
        cartesian = torch.zeros((3, 3, 3, 3), dtype=torch.float64)
        # TODO: figure out short way that use transposes 
        for i in range(1, 3+1):
            for j in range(1, 3+1):
                for k in range(1, 3+1):
                    for n in range(1, 3+1):
                        p = (i, j) if i <= j else (j, i)
                        s = (k, n) if k <= n else (n, k)
                        cartesian[i-1, j-1, k-1, n-1] = voigt[voigt_map[s]-1, voigt_map[p]-1]

        self.cartesian = cartesian
        return self.cartesian 

    def cartesian_to_voigt(self):
        cartesian = self.cartesian.clone()
        voigt = torch.zeros((6, 6), dtype=torch.float64)
        
        voigt[0, 0] = 0.5*cartesian[0, 0, 0, 0]
        voigt[0, 1] = cartesian[0, 0, 1, 1]
        voigt[0, 2] = cartesian[0, 0, 2, 2]
        voigt[0, 3] = cartesian[0, 0, 1, 2]
        voigt[0, 4] = cartesian[0, 0, 0, 2]
        voigt[0, 5] = cartesian[0, 0, 0, 1]
        
        voigt[1, 1] = 0.5*cartesian[1, 1, 1, 1]
        voigt[1, 2] = cartesian[1, 1, 2, 2]
        voigt[1, 3] = cartesian[1, 1, 1, 2]
        voigt[1, 4] = cartesian[1, 1, 0, 2]
        voigt[1, 5] = cartesian[1, 1, 0, 1]

        voigt[2, 2] = 0.5*cartesian[2, 2, 2, 2]
        voigt[2, 3] = cartesian[2, 2, 1, 2]
        voigt[2, 4] = cartesian[2, 2, 0, 2]
        voigt[2, 5] = cartesian[2, 2, 0, 1]

        voigt[3, 3] = 0.5*cartesian[1, 2, 1, 2]
        voigt[3, 4] = cartesian[1, 2, 0, 2]
        voigt[3, 5] = cartesian[1, 2, 0, 1]
        
        voigt[4, 4] = 0.5*cartesian[0, 2, 0, 2]
        voigt[4, 5] = cartesian[0, 2, 0, 1]

        voigt[5, 5] = 0.5*cartesian[0, 1, 0, 1]
 
        voigt = voigt + voigt.t()

        self.voigt = voigt
        return self.voigt

    def cartesian_to_covariant(self):
        cartesian = self.cartesian.clone()
        factor = self.cov_factor
        cov_mat = factor * self.cov_mat
        covariant = torch.einsum("ijkn,ia,jb,kc,nd->abcd", cartesian, cov_mat, cov_mat, cov_mat, cov_mat)
        self.covariant = covariant
        return self.covariant

    def covariant_to_cartesian(self):
        covariant = self.covariant.clone()
        factor = 1. / self.cov_factor
        uncov_mat = factor * self.cov_mat
        cartesian = torch.einsum("abcd,ia,jb,kc,nd->ijkn", covariant, uncov_mat, uncov_mat, uncov_mat, uncov_mat)
        self.cartesian = cartesian
        return self.cartesian

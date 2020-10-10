import numpy as np

from elastic_tensor.elastic_coef import deconstruction_coef, reconstruction_coef


class ElasticTensor:
    def __init__(self, data, representation):
        assert representation in ['voigt', 'cartesian', 'covariant', 'spherical'], "Representation of elastic tensor should be either one of the following: 'voigt', 'cartesian', 'covariant' or 'spherical'"
        data = {representation: data}

        self.voigt = data.get('voigt')
        self.cartesian = data.get('cartesian')
        self.covariant = data.get('covariant')
        self.spherical = data.get('spherical')

        self.voigt_map = {(1, 1): 1, (2, 2): 2, (3, 3): 3, (2, 3): 4, (1, 3): 5, (1, 2): 6}
        self.xyz_to_yzx = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        self.yzx_to_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        self.K = -np.sqrt(3./(4.*np.pi)) * self.xyz_to_yzx
        self.J = -np.sqrt(4.*np.pi/3.) * self.yzx_to_xyz
        self.T = np.array(reconstruction_coef().tolist()).astype(np.float64) / (8 * np.power(np.pi, 3./2))
        self.T_inv = np.array(deconstruction_coef().tolist()).astype(np.float64) * (8 * np.power(np.pi, 3./2))
        self.linearly_independent_subset_idx = np.array([0, 1, 2, 4, 5, 8, 10, 11, 13, 14, 17, 20, 22, 23, 26, 40, 41, 44, 50, 53, 80], dtype=np.int64)

    def voigt_to_cartesian(self):
        voigt = self.voigt
        voigt_map = self.voigt_map
        cartesian = np.zeros((3, 3, 3, 3), dtype=np.float64)
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
        cartesian = self.cartesian
        voigt = np.zeros((6, 6), dtype=np.float64)
        
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

        self.voigt = voigt + voigt.T
        return self.voigt

    def cartesian_to_covariant(self):
        K = self.K
        self.covariant = np.einsum("ijkn,ai,bj,ck,dn->abcd", self.cartesian, K, K, K, K)
        return self.covariant

    def covariant_to_cartesian(self):
        J = self.J
        self.cartesian = np.einsum("abcd,ia,jb,kc,nd->ijkn", self.covariant, J, J, J, J)
        return self.cartesian

    def covariant_to_spherical(self):
        C_hat = self.covariant.reshape(-1)[self.linearly_independent_subset_idx]
        self.spherical = self.T_inv @ C_hat
        return self.spherical  # [P_00, P_2m, S_00, S_2m, S_4m]

    def spherical_to_covariant(self):
        covariant = np.zeros(81, dtype=np.float64)

        C_hat = self.T @ self.spherical  # [21]
        covariant[self.linearly_independent_subset_idx] = C_hat
        covariant = covariant.reshape((3, 3, 3, 3))

        # (a, b, c, d) = (c, d, a, b)
        transpose = covariant.transpose(2, 3, 0, 1).copy()
        covariant[transpose != 0.] = transpose[transpose != 0.]

        # (a, b, c, d) = (b, a, c, d)
        transpose = covariant.transpose(1, 0, 2, 3).copy()
        covariant[transpose != 0.] = transpose[transpose != 0.]

        # (a, b, c, d) = (a, b, d, c)
        transpose = covariant.transpose(0, 1, 3, 2).copy()
        covariant[transpose != 0.] = transpose[transpose != 0.]

        self.covariant = covariant
        return self.covariant

import numpy as np
from elastic_tensor.elastic_tensor import ElasticTensor
from e3nn.o3 import rand_angles
from e3nn.o3 import irr_repr

data = np.random.randn(36).reshape((6,6))
et1 = ElasticTensor(data, 'voigt')

# back and forth transformation to cartesian ensures that voigt is symmetric
_ = et1.voigt_to_cartesian()
_ = et1.cartesian_to_voigt()

_ = et1.voigt_to_cartesian()
_ = et1.cartesian_to_covariant()
_ = et1.covariant_to_spherical()

et2 = ElasticTensor(et1.covariant.copy(), 'covariant')

a, b, c = rand_angles()

D0 = irr_repr(0, a, b, c).numpy()
D1 = irr_repr(1, a, b, c).numpy()
D2 = irr_repr(2, a, b, c).numpy()
D4 = irr_repr(4, a, b, c).numpy()

# forward rotation
et2.covariant = np.einsum('ijkn,ai,bj,ck,dn->abcd', et2.covariant, D1, D1, D1, D1)
_ = et2.covariant_to_spherical()

# inverse rotation 
et2.spherical[0:1] = D0.T @ et2.spherical[0:1]
et2.spherical[1:6] = D2.T @ et2.spherical[1:6]
et2.spherical[6:7] = D0.T @ et2.spherical[6:7]
et2.spherical[7:12] = D2.T @ et2.spherical[7:12]
et2.spherical[12:21] = D4.T @ et2.spherical[12:21]

_ = et2.spherical_to_covariant()
_ = et2.covariant_to_cartesian()
_ = et2.cartesian_to_voigt()

assert np.allclose(et1.spherical, et2.spherical), f"covariant -> spherical differs from covariant -> rotation -> spherical -> inverse rotation by up to {np.abs(et1.spherical - et2.spherical).max()}"

assert np.allclose(et1.covariant, et2.covariant), f"covariant differs from spherical -> covariant"
assert np.allclose(et1.cartesian, et2.cartesian), f"cartesian differs from covariant -> cartesian"
assert np.allclose(et1.voigt, et2.voigt), f"voigt differs from cartesian -> voigt"


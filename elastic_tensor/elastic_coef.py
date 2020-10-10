import sympy
from sympy import sqrt

from elastic_tensor.rgaunt import rgaunt_p as rgaunt

from functools import partial

common = rgaunt(0, 0, 0, 0, 0, 0)

rgaunt_011 = partial(rgaunt, l1=0, l2=1, l3=1)
rgaunt_211 = partial(rgaunt, l1=2, l2=1, l3=1)

rgaunt_022 = partial(rgaunt, l1=0, l2=2, l3=2)
rgaunt_222 = partial(rgaunt, l1=2, l2=2, l3=2)
rgaunt_422 = partial(rgaunt, l1=4, l2=2, l3=2)


def A(alpha, beta, gamma, delta):
    return rgaunt_011(0, alpha, beta) * rgaunt_011(0, gamma, delta) * common


def B(alpha, beta, gamma, delta, m):
    B_value = rgaunt_022(0, m, m) * (rgaunt_011(0, alpha, beta) * rgaunt_211(m, gamma, delta) + rgaunt_011(0, gamma, delta) * rgaunt_211(m, alpha, beta))
    return B_value


def D(alpha, beta, gamma, delta):
    quintet = [-2, -1, 0, +1, +2]
    D_value = 0
    for mp in quintet:
        D_value += rgaunt_211(mp, alpha, beta) * rgaunt_211(mp, gamma, delta) * rgaunt_022(0, mp, mp)

    return D_value


def E(alpha, beta, gamma, delta, m):
    quintet = [-2, -1, 0, +1, +2]
    E_value = 0
    for m1 in quintet:
        for m2 in quintet:
            E_value += rgaunt_211(m1, alpha, beta) * rgaunt_211(m2, gamma, delta) * rgaunt_222(m, m1, m2)

    return E_value


def H(alpha, beta, gamma, delta, m):
    quintet = [-2, -1, 0, +1, +2]
    H_value = 0
    for m1 in quintet:
        for m2 in quintet:
            H_value += rgaunt_211(m1, alpha, beta) * rgaunt_211(m2, gamma, delta) * rgaunt_422(m, m1, m2)

    return H_value


def sph_cov_mat():
    B_n2 = partial(B, m=-2)
    B_n1 = partial(B, m=-1)
    B_0 = partial(B, m=0)
    B_p1 = partial(B, m=+1)
    B_p2 = partial(B, m=+2)

    E_n2 = partial(E, m=-2)
    E_n1 = partial(E, m=-1)
    E_0 = partial(E, m=0)
    E_p1 = partial(E, m=+1)
    E_p2 = partial(E, m=+2)

    H_n4 = partial(H, m=-4)
    H_n3 = partial(H, m=-3)
    H_n2 = partial(H, m=-2)
    H_n1 = partial(H, m=-1)
    H_0 = partial(H, m=0)
    H_p1 = partial(H, m=+1)
    H_p2 = partial(H, m=+2)
    H_p3 = partial(H, m=+3)
    H_p4 = partial(H, m=+4)

    sph_func = [A,
                B_n2, B_n1, B_0, B_p1, B_p2,
                D,
                E_n2, E_n1, E_0, E_p1, E_p2,
                H_n4, H_n3, H_n2, H_n1, H_0, H_p1, H_p2, H_p3, H_p4]

    mat = sympy.zeros(81, 21)

    triplet = [-1, 0, +1]
    for idx in range(81):
        alpha_p = idx // 27
        beta_p = (idx % 27) // 9
        gamma_p = (idx % 9) // 3
        delta_p = idx % 3

        alpha = alpha_p - 1
        beta = beta_p - 1
        gamma = gamma_p - 1
        delta = delta_p - 1
        for j, sph in enumerate(sph_func):
            mat[idx, j] = sph(alpha, beta, gamma, delta)

    return mat


def sph_cov_mat_sub():
    B_n2 = partial(B, m=-2)
    B_n1 = partial(B, m=-1)
    B_0 = partial(B, m=0)
    B_p1 = partial(B, m=+1)
    B_p2 = partial(B, m=+2)

    E_n2 = partial(E, m=-2)
    E_n1 = partial(E, m=-1)
    E_0 = partial(E, m=0)
    E_p1 = partial(E, m=+1)
    E_p2 = partial(E, m=+2)

    H_n4 = partial(H, m=-4)
    H_n3 = partial(H, m=-3)
    H_n2 = partial(H, m=-2)
    H_n1 = partial(H, m=-1)
    H_0 = partial(H, m=0)
    H_p1 = partial(H, m=+1)
    H_p2 = partial(H, m=+2)
    H_p3 = partial(H, m=+3)
    H_p4 = partial(H, m=+4)

    sph_func = [A,
                B_n2, B_n1, B_0, B_p1, B_p2,
                D,
                E_n2, E_n1, E_0, E_p1, E_p2,
                H_n4, H_n3, H_n2, H_n1, H_0, H_p1, H_p2, H_p3, H_p4]

    mat = sympy.zeros(21, 21)

    quadruplets = [(-1, -1, -1, -1), (-1, -1, -1, 0), (-1, -1, -1, +1), (-1, -1, 0, 0), (-1, -1, 0, +1), (-1, -1, +1, +1),
                   (-1, 0, -1, 0), (-1, 0, -1, +1), (-1, 0, 0, 0), (-1, 0, 0, +1), (-1, 0, +1, +1),
                   (-1, +1, -1, +1), (-1, +1, 0, 0), (-1, +1, 0, +1), (-1, +1, +1, +1),
                   (0, 0, 0, 0), (0, 0, 0, +1), (0, 0, +1, +1),
                   (0, +1, 0, +1), (0, +1, +1, +1),
                   (+1, +1, +1, +1)]

    for idx, quadruplet in enumerate(quadruplets):
        for j, sph in enumerate(sph_func):
            mat[idx, j] = sph(*quadruplet)

    return mat


def reconstruction_coef():
    T_scaled = sympy.Matrix([
        [1, 0, 0, -2 * sqrt(5) / 5, 0, -2 * sqrt(15) / 5, 4 / 5, 0, 0, -4 * sqrt(5) / 35, 0, -4 * sqrt(15) / 35, 0, 0, 0, 0, 9 / 35, 0, 6 * sqrt(5) / 35, 0, 3 * sqrt(35) / 35],
        [0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, -3 * sqrt(70) / 70, 0, -9 * sqrt(10) / 70, 0, 0, 0, 0, 0],
        [0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, -3 * sqrt(35) / 35, 0, -3 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, sqrt(5) / 5, 0, -sqrt(15) / 5, -2 / 5, 0, 0, -4 * sqrt(5) / 35, 0, 4 * sqrt(15) / 35, 0, 0, 0, 0, -12 / 35, 0, -6 * sqrt(5) / 35, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, -4 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, -3 * sqrt(10) / 70, 0, -3 * sqrt(70) / 70, 0],
        [1, 0, 0, -2 * sqrt(5) / 5, 0, 0, -2 / 5, 0, 0, 8 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0, 3 / 35, 0, 0, 0, -3 * sqrt(35) / 35],
        [0, 0, 0, 0, 0, 0, 3 / 5, 0, 0, 3 * sqrt(5) / 35, 0, -3 * sqrt(15) / 35, 0, 0, 0, 0, -12 / 35, 0, -6 * sqrt(5) / 35, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, -3 * sqrt(10) / 70, 0, -3 * sqrt(70) / 70, 0],
        [0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, 6 * sqrt(10) / 35, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, 6 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0],
        [0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, -4 * sqrt(15) / 35, 0, 0, 0, 0, 3 * sqrt(70) / 70, 0, -3 * sqrt(10) / 70, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3 / 5, 0, 0, -6 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0, 3 / 35, 0, 0, 0, -3 * sqrt(35) / 35],
        [0, sqrt(15) / 5, 0, 0, 0, 0, 0, -4 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, 6 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3 * sqrt(15) / 35, 0, 0, 0, 0, 3 * sqrt(70) / 70, 0, -3 * sqrt(10) / 70, 0, 0, 0, 0, 0],
        [0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, 3 * sqrt(35) / 35, 0, -3 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 4 * sqrt(5) / 5, 0, 0, 4 / 5, 0, 0, 8 * sqrt(5) / 35, 0, 0, 0, 0, 0, 0, 24 / 35, 0, 0, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, 6 * sqrt(10) / 35, 0, 0, 0],
        [1, 0, 0, sqrt(5) / 5, 0, sqrt(15) / 5, -2 / 5, 0, 0, -4 * sqrt(5) / 35, 0, -4 * sqrt(15) / 35, 0, 0, 0, 0, -12 / 35, 0, 6 * sqrt(5) / 35, 0, 0],
        [0, 0, 0, 0, 0, 0, 3 / 5, 0, 0, 3 * sqrt(5) / 35, 0, 3 * sqrt(15) / 35, 0, 0, 0, 0, -12 / 35, 0, 6 * sqrt(5) / 35, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 5, 0, 0, 0, 0, 0, 2 * sqrt(15) / 35, 0, 0, 0, 0, 0, 0, -9 * sqrt(10) / 70, 0, 3 * sqrt(70) / 70, 0],
        [1, 0, 0, -2 * sqrt(5) / 5, 0, 2 * sqrt(15) / 5, 4 / 5, 0, 0, -4 * sqrt(5) / 35, 0, 4 * sqrt(15) / 35, 0, 0, 0, 0, 9 / 35, 0, -6 * sqrt(5) / 35, 0, 3 * sqrt(35) / 35]
    ])
    return T_scaled


def deconstruction_coef():
    T_inv_scaled_transposed = sympy.Matrix([
        [1 / 9, 0, 0, -sqrt(5) / 18, 0, -sqrt(15) / 18, 1 / 9, 0, 0, -sqrt(5) / 18, 0, -sqrt(15) / 18, 0, 0, 0, 0, 1 / 8, 0, sqrt(5) / 12, 0, sqrt(35) / 24],
        [0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, -sqrt(70) / 12, 0, -sqrt(10) / 4, 0, 0, 0, 0, 0],
        [0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, -sqrt(35) / 6, 0, -sqrt(5) / 6, 0, 0, 0, 0, 0, 0],
        [2 / 9, 0, 0, sqrt(5) / 18, 0, -sqrt(15) / 18, -1 / 9, 0, 0, -sqrt(5) / 9, 0, sqrt(15) / 9, 0, 0, 0, 0, -1 / 3, 0, -sqrt(5) / 6, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, -2 * sqrt(15) / 9, 0, 0, 0, 0, 0, 0, -sqrt(10) / 12, 0, -sqrt(70) / 12, 0],
        [2 / 9, 0, 0, -sqrt(5) / 9, 0, 0, -1 / 9, 0, 0, 2 * sqrt(5) / 9, 0, 0, 0, 0, 0, 0, 1 / 12, 0, 0, 0, -sqrt(35) / 12],
        [0, 0, 0, 0, 0, 0, 1 / 3, 0, 0, sqrt(5) / 6, 0, -sqrt(15) / 6, 0, 0, 0, 0, -2 / 3, 0, -sqrt(5) / 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt(15) / 3, 0, 0, 0, 0, 0, 0, -sqrt(10) / 6, 0, -sqrt(70) / 6, 0],
        [0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, 0, sqrt(10) / 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, sqrt(15) / 3, 0, 0, 0, 0, 0, 0, 2 * sqrt(5) / 3, 0, 0, 0, 0, 0, 0],
        [0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, -2 * sqrt(15) / 9, 0, 0, 0, 0, sqrt(70) / 12, 0, -sqrt(10) / 12, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1 / 3, 0, 0, -sqrt(5) / 3, 0, 0, 0, 0, 0, 0, 1 / 6, 0, 0, 0, -sqrt(35) / 6],
        [0, sqrt(15) / 9, 0, 0, 0, 0, 0, -2 * sqrt(15) / 9, 0, 0, 0, 0, 0, 0, sqrt(5) / 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, sqrt(15) / 3, 0, 0, 0, 0, sqrt(70) / 6, 0, -sqrt(10) / 6, 0, 0, 0, 0, 0],
        [0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, sqrt(35) / 6, 0, -sqrt(5) / 6, 0, 0, 0, 0, 0, 0],
        [1 / 9, 0, 0, sqrt(5) / 9, 0, 0, 1 / 9, 0, 0, sqrt(5) / 9, 0, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, 0, sqrt(10) / 3, 0, 0, 0],
        [2 / 9, 0, 0, sqrt(5) / 18, 0, sqrt(15) / 18, -1 / 9, 0, 0, -sqrt(5) / 9, 0, -sqrt(15) / 9, 0, 0, 0, 0, -1 / 3, 0, sqrt(5) / 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 1 / 3, 0, 0, sqrt(5) / 6, 0, sqrt(15) / 6, 0, 0, 0, 0, -2 / 3, 0, sqrt(5) / 3, 0, 0],
        [0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, sqrt(15) / 9, 0, 0, 0, 0, 0, 0, -sqrt(10) / 4, 0, sqrt(70) / 12, 0],
        [1 / 9, 0, 0, -sqrt(5) / 18, 0, sqrt(15) / 18, 1 / 9, 0, 0, -sqrt(5) / 18, 0, sqrt(15) / 18, 0, 0, 0, 0, 1 / 8, 0, -sqrt(5) / 12, 0, sqrt(35) / 24]
    ])
    return T_inv_scaled_transposed.T

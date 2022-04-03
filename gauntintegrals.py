import numpy as np
import scipy.special
import scipy.integrate

def compute_real_sph_harm(L, phi, theta):
    (l, m) = L

    Y = scipy.special.sph_harm(abs(m), l, phi, theta)

    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real

    return Y

def compute_gaunt_integral(a, b, c):
    def f(theta, phi):
        return (
            compute_real_sph_harm(a, phi, theta)
            * compute_real_sph_harm(b, phi, theta)
            * compute_real_sph_harm(c, phi, theta)
            * np.sin(theta)
        )

    result, abserror = scipy.integrate.dblquad(
        f,
        0,
        2 * np.pi,
        lambda phi: 0,
        lambda phi: np.pi
    )

    return result

LMAX = 2

matrix_indices = []
matrix = np.zeros((9, 9, 9))

for l in range(0, LMAX + 1):
    for m in range(-l, l + 1):
        matrix_indices.append((l, m))

for i, a in enumerate(matrix_indices):
    for j, b in enumerate(matrix_indices):
        for k, c in enumerate(matrix_indices):
            I = compute_gaunt_integral(a, b, c)
            if I < 1e-15: I = 0
            print(I)
            matrix[i, j, k] = I

np.save("cg-coefficients", matrix)

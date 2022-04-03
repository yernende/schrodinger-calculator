import numpy as np
import scipy.special
import scipy.linalg

from src.classes import Atom
from src.classes import Vector
from src.radial_part import compute_radial_part
from src.gaunt_integral import compute_real_sph_harm

from matplotlib import pyplot

LMAX = 2
cg_coefficients = np.load("cg-coefficients.npy")

atoms = [
    Atom([0, 0, 0], atomic_number=1, radius_muffintin=1)
]

def modified_spherical_hankel_function(order, x):
    if order == 0:
        return np.exp(-x) / x
    elif order == 1:
        return -np.exp(-x) * (x + 1) / x ** 2
    elif order == 2:
        return np.exp(-x) * (x ** 2 + 3 * x + 3) / x ** 3

def modified_spherical_hankel_function_derivative(order, x):
    if order == 0:
        return -np.exp(-x) * (x + 1) / x ** 2
    elif order == 1:
        return np.exp(-x) * (x * (x + 2) + 2) / x ** 3
    elif order == 2:
        return -np.exp(-x) / x**4 * (x * (x * (x + 4) + 9) + 9)

determinants = []
energies = np.linspace(-0.8, -1.5, 100)

matrix_indices = []

for p in range(len(atoms)):
    for l in range(0, LMAX + 1):
        for m in range(-l, l + 1):
            matrix_indices.append((p, l, m))

for energy in energies:
    diagonal_elements = []

    potential = 0
    k = np.sqrt(np.abs(energy - potential))

    for atom in atoms:
        for l in range(0, LMAX + 1):
            R_normalized = compute_radial_part(energy, l, atom.atomic_number, atom.radius_muffintin)

            if energy > potential:
                j1 = scipy.special.spherical_jn(l, k * atom.radius_muffintin)
                f1 = scipy.special.spherical_yn(l, k * atom.radius_muffintin)

                j1_derivative = scipy.special.spherical_jn(l, k * atom.radius_muffintin, True)
                f1_derivative = scipy.special.spherical_yn(l, k * atom.radius_muffintin, True)
            else:
                j1 = scipy.special.spherical_in(l, k * atom.radius_muffintin)
                f1 = modified_spherical_hankel_function(l, k * atom.radius_muffintin)

                j1_derivative = scipy.special.spherical_in(l, k * atom.radius_muffintin, True)
                f1_derivative = modified_spherical_hankel_function_derivative(l, k * atom.radius_muffintin)

            t = -(
                (j1 * R_normalized[2][-1] - j1_derivative * R_normalized[1][-1])
                / (f1 * R_normalized[2][-1] - f1_derivative * R_normalized[1][-1])
            )

            for m in range(-l, l + 1):
                diagonal_elements.append(1.0 / t)

    matrix = np.diag(diagonal_elements)

    for i, (p1, l1, m1) in enumerate(matrix_indices):
        for j, (p2, l2, m2) in enumerate(matrix_indices):
            if p1 == p2: continue

            interatomic_vector = Vector(atoms[p2].position.cartesian - atoms[p1].position.cartesian)

            if energy > potential:
                result = 0

                for l in range(0, LMAX + 1):
                    for m in range(-l, l + 1):
                        result += (
                            1j ** (-l)
                            * cg_coefficients[
                                l ** 2 + l + m,
                                l1 ** 2 + l1 + m1,
                                l2 ** 2 + l2 + m2

                            ]
                            * scipy.special.spherical_yn(l, k * interatomic_vector.module)
                            * compute_real_sph_harm((l, m), interatomic_vector.phi, interatomic_vector.theta)
                        )

                result *= (-4 * np.pi * 1j ** (l1 - l2))
                matrix[i, j] = result.real
            else:
                result = 0

                for l in range(0, LMAX + 1):
                    for m in range(-l, l + 1):
                        result += (
                            cg_coefficients[
                                l ** 2 + l + m,
                                l1 ** 2 + l1 + m1,
                                l2 ** 2 + l2 + m2
                            ]
                            * modified_spherical_hankel_function(l, k * interatomic_vector.module)
                            * compute_real_sph_harm((l, m), interatomic_vector.phi, interatomic_vector.theta)
                        )

                result *= (4 * np.pi * (-1) ** (l1 + l2))
                matrix[i, j] = result.real

    determinants.append(scipy.linalg.det(matrix))

for determinant in determinants:
    print(determinant)

pyplot.plot(energies, determinants)
pyplot.show()

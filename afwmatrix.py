import numpy as np
import scipy.special

from src.constants import *
from src.classes import Vector

def compute_F(atom, wave_i, wave_j, energy):
    F = np.dot(wave_i.cartesian, wave_j.cartesian) - energy

    if wave_i == wave_j or np.array_equal(wave_i.cartesian, wave_j.cartesian):
        F *= 1/3
    else:
        F *= (
            scipy.special.spherical_jn(1, Vector.get_distance_between(wave_i, wave_j) * atom.radius_muffintin)
            / Vector.get_distance_between(wave_i, wave_j)
        )

    l_array = np.array(range(MAX_L))

    F -= (
        (2 * l_array + 1)
        * scipy.special.eval_legendre(l_array, Vector.get_cos_of_angle_between(wave_i, wave_j))
        * scipy.special.spherical_jn(l_array, wave_i.module * atom.radius_muffintin)
        * scipy.special.spherical_jn(l_array, wave_j.module * atom.radius_muffintin)
        * atom.radial_part_derivative_to_itself_ratio_array
    ).sum()

    return F

def compute_matrix(atoms, waves, energy):
    matrix = np.zeros((len(waves),) * 2, dtype=complex)

    for i in range(len(waves)):
        for j in range(len(waves)):
            if i > j:
                continue
            elif i == j:
                matrix[i, j] = np.dot(waves[i].cartesian, waves[j].cartesian) - energy

            for atom in atoms:
                matrix[i, j] -= (
                    3 * atom.radius_muffintin ** 2 / atom.radius ** 3
                    * np.exp(1j * np.dot(waves[i].cartesian - waves[j].cartesian, atom.position.cartesian))
                    * compute_F(atom, waves[i], waves[j], energy)
                )

            matrix[j, i] = np.conj(matrix[i, j])

    return matrix

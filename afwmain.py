import os
from multiprocessing import Pool

import numpy as np
import scipy.linalg
from matplotlib import pyplot

from src.classes import Atom
from src.classes import Vector
from src.matrix import compute_matrix
from src.radial_part import compute_radial_part

MAX_L = 10

atoms = [Atom([0, 0, 0], 1, radius_muffintin=1.0)]
energies = np.linspace(0.9, 1.1, 100)
vectors = [Vector([t, t, t]) for t in np.linspace(0.5, 1.5, 11)]

def compute_for_fixed_energy(energy):
    results_for_fixed_energy = []

    for atom in atoms:
        atom.radial_part_derivative_to_itself_ratio_array = np.zeros(MAX_L)

        for l in range(MAX_L):
            R = compute_radial_part(energy, l, atomic_number=atom.atomic_number, radius_max=atom.radius_muffintin)
            atom.radial_part_derivative_to_itself_ratio_array[l] = (
                R[2][-1]
                / R[1][-1]
            )

    matrix = compute_matrix(atoms, vectors, energy)
    determinant = scipy.linalg.det(np.abs(matrix))

    results_for_fixed_energy.append({
        'energy': energy,
        'determinant': determinant
    })

    print(results_for_fixed_energy[-1])
    return results_for_fixed_energy

if __name__ == '__main__':
    with Pool(os.cpu_count()) as pool:
        results = []

        for results_for_fixed_energy in pool.imap_unordered(
            compute_for_fixed_energy,
            energies,
            int(len(energies) / os.cpu_count())
        ):
            results += results_for_fixed_energy

        print('\nFinal result')
        print(min(results, key=lambda result: result['determinant']))

        X, Y = [], []

        for result in sorted(results, key=lambda result: result['energy']):
            X.append(result['energy'])
            Y.append(result['determinant'])

        X, Y = np.array(X), np.array(Y)

        pyplot.ylabel("Детерминант модуля матрицы")
        pyplot.xlabel("Энергия атома")
        pyplot.plot(X, Y)
        pyplot.show()

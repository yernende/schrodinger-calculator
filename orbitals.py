import ctypes
import pathlib
import csv

import numpy as np
from numpy import savetxt
from numpy import loadtxt
import scipy.interpolate
import scipy.integrate
import scipy.special

NMAX = 6
LMAX = 2

data = loadtxt('data/potentials.csv', delimiter=',')
potentials_x = np.array(data[:,0])
potentials_y = np.array(data[:,1])

coefficients = np.zeros((NMAX, (LMAX + 1), (LMAX * 2 + 1)))
coefficients_packed = np.empty(NMAX * (LMAX + 1) * (LMAX * 2 + 1))

with open('data/coefficients.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if int(row[1]) == 1:
            coefficients[int(row[0])-1, int(row[2]), int(row[3])] = float(row[4])

coefficient_index = 0
for n in range(1, NMAX+1):
    for l in range(0, (LMAX + 1)):
        for m in range(-l, l + 1):
            coefficients_packed[coefficient_index] = coefficients[n-1, l, m]
            coefficient_index += 1

energies = np.array([
    -2.958291,
    -2.675457,
    -2.295286,
    -1.976262,
    -1.847000,
    -1.783929
])

liborbitals = ctypes.CDLL(pathlib.Path().absolute() / "liborbitals.so")

atomic_number = 6
radius_max = 1.663942

grid_width = 301
grid_packed = np.empty(grid_width ** 3)

liborbitals.fill_grid(
    energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(atomic_number),
    ctypes.c_double(radius_max),
    grid_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(grid_width),
    coefficients_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    potentials_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    potentials_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(potentials_x.size)
)

print("Reshaping...", end='', flush=True)
grid = grid_packed.reshape(grid_width, grid_width, grid_width)
print(" done")

print("Importing MayaVi...", end='', flush=True)
from mayavi import mlab
print(" done")

mlab.contour3d(grid)
mlab.show()

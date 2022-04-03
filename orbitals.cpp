#include <complex>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdexcept>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>

#include "coords.h"
#include "numerov.cpp"

const std::complex<double> IMAGINARY_UNIT = {0, 1};

#define grid_get(grid, i, j, k) (grid[grid_width * grid_width * i + grid_width * j + k])

extern "C" void fill_grid(
  double *energies, int atomic_number, double radius_max, double *grid, int grid_width, double *coefficients,
  double *potential_x, double *potential_y, int potential_points_amount
) {
  if (grid_width % 2 == 0) {
    throw std::invalid_argument("grid_width should be an odd integer");
  }

  int grid_width_half = (grid_width + 1) / 2;

  const int NMAX = 6;
  const int LMAX = 2;

  int radial_part_points_amount = 1000;
  double P[NMAX][LMAX + 1][3][radial_part_points_amount];
  double grid_scale = radius_max / ((grid_width_half - 1));

  std::cout << "Preparing potential interpolator..." << std::flush;
  double *potential_without_doubles_x = new double[potential_points_amount];
  double *potential_without_doubles_y = new double[potential_points_amount];
  int j = 0;

  for (int i = 0; i < potential_points_amount; i++) {
    if (i > 0 && potential_x[i] != potential_x[i - 1]) {
      potential_without_doubles_x[j] = potential_x[i];
      potential_without_doubles_y[j] = potential_y[i];
      j++;
    }
  }

  int potential_without_doubles_points_amount = j;

  InterpolationData potential_interpolation_data = {
    gsl_interp_accel_alloc(),
    gsl_spline_alloc(gsl_interp_cspline, potential_without_doubles_points_amount)
  };

  gsl_spline_init(
    potential_interpolation_data.spline, potential_without_doubles_x, potential_without_doubles_y,
    potential_without_doubles_points_amount
  );

  potential_interpolation_data.min_x = potential_without_doubles_x[0];
  potential_interpolation_data.max_x = potential_without_doubles_x[potential_without_doubles_points_amount - 1];

  delete [] potential_without_doubles_x;
  delete [] potential_without_doubles_y;

  std::cout << " done" << std::endl;

  std::cout << "Computing radial parts..." << std::flush;
  for (int n = 1; n <= NMAX; n++) {
    for (int l = 0; l <= LMAX; l++) {
      compute_radial_part_numerov_normalized(
        P[n-1][l][0], P[n-1][l][1], P[n-1][l][2], energies[n-1], l, atomic_number, radius_max, radial_part_points_amount,
        &potential_interpolation_data
      );
    }
  }
  std::cout << " done" << std::endl;

  gsl_spline_free(potential_interpolation_data.spline);
  gsl_interp_accel_free(potential_interpolation_data.acc);

  std::cout << "Computing grid..." << std::flush;
  for (int i = 0; i < grid_width; i++) {
    for (int j = 0; j < grid_width; j++) {
      for (int k = 0; k < grid_width; k++) {
        coords_cartesian point_cartesian = {
          (double) (i - (grid_width_half - 1)) * grid_scale,
          (double) (j - (grid_width_half - 1)) * grid_scale,
          (double) (k - (grid_width_half - 1)) * grid_scale
        };

        coords_spherical point_spherical = point_cartesian.to_spherical();

        if (point_spherical.r > radius_max) {
          continue;
        }

        grid_get(grid, i, j, k) = 0;

        int coefficient_index = 0;

        double legendre_polynomials[gsl_sf_legendre_array_n(LMAX)];
        gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM, LMAX, cos(point_spherical.theta), 1, legendre_polynomials);

        for (int n = 1; n <= NMAX; n++) {
          for (int l = 0; l <= LMAX; l++) {
            for (int m = -l; m <= l; m++) {
              std::complex<double> spherical_function =
                std::exp(IMAGINARY_UNIT * (double) abs(m) * point_spherical.phi)
                * legendre_polynomials[gsl_sf_legendre_array_index(l, abs(m))];

              if (m < 0) {
                spherical_function = std::conj(spherical_function) * std::pow(-1, m);
              }

              grid_get(grid, i, j, k) += std::pow(
                std::abs(
                  coefficients[coefficient_index]
                  * spherical_function
                  * P[n-1][l][1][(int) std::round((radial_part_points_amount - 1) * (point_spherical.r / radius_max))]
                ), 1
              );

              coefficient_index++;
            }
          }
        }
      }
    }
  }
  std::cout << " done" << std::endl;
}

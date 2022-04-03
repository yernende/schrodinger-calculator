#include <iostream>

#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>

struct InterpolationData {
  gsl_interp_accel *acc;
  gsl_spline *spline;
  double min_x;
  double max_x;
};

double inline static V(
  double radius, double energy, int l, int atomic_number, InterpolationData *potential_interpolation_data
) {
  double result;

  if (radius == 0) {
    result = 0;
  } else {
    result = l * (l + 1) / (radius * radius) - energy;

    if (radius > potential_interpolation_data->min_x && radius < potential_interpolation_data->max_x) {
      result += gsl_spline_eval(potential_interpolation_data->spline, radius, potential_interpolation_data->acc);
    } else {
      result += -2 * (double) atomic_number / radius;
    }
  }

  return result;
}

int compute_radial_part_numerov_dense(
  double energy, int l, int atomic_number, double radius_max, double *&x_dense, double *&y_dense,
  InterpolationData *potential_interpolation_data, double h=1e-4
) {
  int points_amount = radius_max / h + 1;

  double *f = new double[points_amount];
  x_dense = new double[points_amount];
  y_dense = new double[points_amount];

  x_dense[0] = 0;
  y_dense[0] = 0;
  y_dense[1] = h;

  for (int i = 1; i < points_amount; i++) {
    x_dense[i] = x_dense[i - 1] + h;
    f[i] = (1 - (h * h) / 12.0 * V(x_dense[i], energy, l, atomic_number, potential_interpolation_data));
  }

  for (int i = 2; i < points_amount; i++) {
    y_dense[i] = ((12 - 10 * f[i - 1]) * y_dense[i - 1] - f[i - 2] * y_dense[i - 2]) / f[i];
  }

  x_dense[points_amount - 1] = radius_max;

  delete [] f;
  return points_amount;
}

double static radial_part_square(double x, void *p) {
  InterpolationData *params = (InterpolationData *)p;
  return std::pow(gsl_spline_eval(params->spline, x, params->acc), 2);
}

double static radial_part_after_zero(double x, void *p) {
  InterpolationData *params = (InterpolationData *)p;
  return gsl_spline_eval(params->spline, x, params->acc) / x;
}

double compute_normalization_factor(InterpolationData *interpolation_data, double radius_max) {
  gsl_integration_workspace *integration_workspace = gsl_integration_workspace_alloc(1000);

  gsl_function F;
  F.function = &radial_part_square;
  F.params = interpolation_data;

  double integration_result, integration_error;

  gsl_integration_qags(&F, 0, radius_max, 0, 1e-7, 1000, integration_workspace, &integration_result,
    &integration_error);
  gsl_integration_workspace_free(integration_workspace);

  return 1 / std::sqrt(integration_result);
}

void static compute_radial_part_numerov_normalized(
  double *x, double *y, double *derivative, double energy, int l, int atomic_number, double radius_max,
  int points_amount, InterpolationData *potential_interpolation_data
) {
  double *x_dense;
  double *y_dense;

  int dense_points_amount = compute_radial_part_numerov_dense(
    energy, l, atomic_number, radius_max, x_dense, y_dense, potential_interpolation_data
  );

  InterpolationData interpolation_data = {
    gsl_interp_accel_alloc(),
    gsl_spline_alloc(gsl_interp_cspline, dense_points_amount)
  };

  gsl_spline_init(interpolation_data.spline, x_dense, y_dense, dense_points_amount);

  double normalization_factor = compute_normalization_factor(&interpolation_data, radius_max);

  x[0] = 0;
  x[points_amount - 1] = radius_max;

  double step_dense = x_dense[1] - x_dense[0];
  double step = radius_max / (points_amount - 1);

  for (int i = 1; i < (points_amount - 1); i++) {
    x[i] = x[i - 1] + step;
  }

  gsl_function F;
  F.function = &radial_part_after_zero;
  F.params = &interpolation_data;

  double derivative_abserr;

  for (int i = 1; i < points_amount; i++) {
    y[i] = radial_part_after_zero(x[i], &interpolation_data) * normalization_factor;
  }

  for (int i = 0; i < points_amount; i++) {
    derivative[i] = 0;
    if (i <= 2) {
      gsl_deriv_forward(&F, x[i], step_dense, &derivative[i], &derivative_abserr);
    } else if (i > 2 && i < points_amount - 2) {
      gsl_deriv_central(&F, x[i], 2 * step_dense, &derivative[i], &derivative_abserr);
    } else {
      gsl_deriv_backward(&F, x[i], step_dense, &derivative[i], &derivative_abserr);
    }

    derivative[i] *= normalization_factor;
  }

  derivative[0] = derivative[1] - (
    - 25/12 * derivative[1]
    + 4 * derivative[2]
    - 3 * derivative[3]
    + 4/3 * derivative[4]
    - 1/4 * derivative[5]
  );

  y[0] = y[1] - derivative[0] * step;

  gsl_spline_free(interpolation_data.spline);
  gsl_interp_accel_free(interpolation_data.acc);

  delete [] x_dense;
  delete [] y_dense;
}

extern "C" void compute_radial_part(
  double *x, double *y, double *derivative, int radial_part_points_amount,
  double *potential_x, double *potential_y, int potential_points_amount,
  double energy, int l, int atomic_number, double radius_max
) {
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

  std::cout << "Computing radial part for energy=" << energy << " l=" << l << " ..."  << std::flush;
  compute_radial_part_numerov_normalized(
    x, y, derivative, energy, l, atomic_number, radius_max, radial_part_points_amount,
    &potential_interpolation_data
  );
  std::cout << " done" << std::endl;
}

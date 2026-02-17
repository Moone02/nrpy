#include "BHaH_defines.h"

/**
 * @brief Computes the normalization constraint of the 4-vector.
 *
 *         Evaluates the scalar invariant:
 *             C = g_munu v^mu v^nu
 *         where v^mu corresponds to the 4-momentum p^mu stored in the state array.
 *
 *         Expected Value: 0.0
 *
 *         Input:
 *             metric: The metric tensor components at the current location.
 *             f[9]: The state vector.
 *                   f[4] -> v^0 (time component)
 *                   f[5] -> v^1 (x component)
 *                   f[6] -> v^2 (y component)
 *                   f[7] -> v^3 (z component)
 *         Output:
 *             norm_out: The computed value of the constraint.
 */
void normalization_constraint_photon(const metric_struct *restrict metric, const double f[9], double *restrict norm_out) {
  // Unpack 4-momentum p^mu components from f[4]..f[7]
  const double vU0 = f[4];
  const double vU1 = f[5];
  const double vU2 = f[6];
  const double vU3 = f[7];
  const REAL tmp0 = 2 * vU0;
  *norm_out = metric->g4DD00 * ((vU0) * (vU0)) + metric->g4DD01 * tmp0 * vU1 + metric->g4DD02 * tmp0 * vU2 + metric->g4DD03 * tmp0 * vU3 +
              metric->g4DD11 * ((vU1) * (vU1)) + 2 * metric->g4DD12 * vU1 * vU2 + 2 * metric->g4DD13 * vU1 * vU3 + metric->g4DD22 * ((vU2) * (vU2)) +
              2 * metric->g4DD23 * vU2 * vU3 + metric->g4DD33 * ((vU3) * (vU3));
} // END FUNCTION normalization_constraint_photon

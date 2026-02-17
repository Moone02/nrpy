#include "BHaH_defines.h"

/**
 * @brief Computes the 10 unique components of the KerrSchild_Cartesian metric for a photon particle.
 */
void g4DD_metric_KerrSchild_Cartesian(const commondata_struct *restrict commondata, const double f[9], metric_struct *restrict metric) {
#include "set_CodeParameters.h"
  // Unpack position coordinates from f[0]..f[3] (State vector size: 9)
  const double x = f[1];
  const double y = f[2];
  const double z = f[3];

  const REAL tmp0 = ((a_spin) * (a_spin));
  const REAL tmp4 = ((z) * (z));
  const REAL tmp6 = (1.0 / 2.0) * tmp4 + (1.0 / 2.0) * ((x) * (x)) + (1.0 / 2.0) * ((y) * (y)) +
                    (1.0 / 2.0) * sqrt(4 * tmp0 * tmp4 + ((-tmp0 + tmp4 + ((x) * (x)) + ((y) * (y))) * (-tmp0 + tmp4 + ((x) * (x)) + ((y) * (y)))));
  const REAL tmp7 = -1.0 / 2.0 * tmp0 + tmp6;
  const REAL tmp12 = (1.0 / 2.0) * tmp0 + tmp6;
  const REAL tmp8 = 2 * M_scale / (tmp0 * tmp4 + ((tmp7) * (tmp7)));
  const REAL tmp10 = sqrt(tmp7);
  const REAL tmp13 = (1.0 / (tmp12));
  const REAL tmp9 = pow(tmp7, 3.0 / 2.0) * tmp8;
  const REAL tmp11 = a_spin * y + tmp10 * x;
  const REAL tmp15 = -a_spin * x + tmp10 * y;
  const REAL tmp16 = tmp7 * tmp8 * z;
  const REAL tmp17 = tmp9 / ((tmp12) * (tmp12));
  metric->g4DD00 = tmp9 - 1;
  metric->g4DD01 = tmp11 * tmp13 * tmp9;
  metric->g4DD02 = tmp13 * tmp15 * tmp9;
  metric->g4DD03 = tmp16;
  metric->g4DD11 = ((tmp11) * (tmp11)) * tmp17 + 1;
  metric->g4DD12 = tmp11 * tmp15 * tmp17;
  metric->g4DD13 = tmp11 * tmp13 * tmp16;
  metric->g4DD22 = ((tmp15) * (tmp15)) * tmp17 + 1;
  metric->g4DD23 = tmp13 * tmp15 * tmp16;
  metric->g4DD33 = tmp10 * tmp4 * tmp8 + 1;
} // END FUNCTION g4DD_metric_KerrSchild_Cartesian

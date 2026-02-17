"""
Generates the C adaptive step-size controller, updated for BHaH integration.

Author: Dalton J. Moone
"""

from nrpy.infrastructures.BHaH import BHaH_defines_h as Bdefines_h

def rkf45_update_and_control_helper() -> None:
    """
    Generate the GSL-style adaptive controller.
    """
    c_code_for_header = r"""
static inline bool update_photon_state_and_stepsize(
    PhotonState *restrict photon,
    const double f_start[9],
    const double f_out[9],
    const double f_err[9],
    const commondata_struct *restrict commondata
) {
    const double rtol = commondata->rkf45_error_tolerance;
    const double atol = commondata->rkf45_absolute_error_tolerance;
    double err_norm_sq = 0.0;

    // Weighted RMS norm calculation
    for (int i = 1; i < 8; ++i) {
        double scale = atol + rtol * fabs(f_start[i]);
        double ratio = f_err[i] / scale;
        err_norm_sq += ratio * ratio;
    }
    // Handle time and path length with absolute tolerance
    double ratio_t = f_err[0] / atol;
    double ratio_L = f_err[8] / atol;
    err_norm_sq += (ratio_t * ratio_t + ratio_L * ratio_L);

    const double err_norm = sqrt(err_norm_sq / 9.0);
    bool step_accepted = (err_norm <= 1.0);

    // Compute next h using 1/5 power law
    double h_new;
    if (err_norm > 1e-15) {
        h_new = commondata->rkf45_safety_factor * photon->h * pow(1.0 / err_norm, 0.2);
    } else {
        h_new = 2.0 * photon->h;
    }

    h_new = fmax(h_new, commondata->rkf45_h_min);
    h_new = fmin(h_new, commondata->rkf45_h_max);

    if (step_accepted) {
        for (int i = 0; i < 9; ++i) photon->f[i] = f_out[i];
        photon->affine_param += photon->h;
        photon->rejection_retries = 0;
    } else {
        photon->rejection_retries++;
    }

    photon->h = h_new;
    return step_accepted;
}
"""
    Bdefines_h.register_BHaH_defines("rkf45_update_control", c_code_for_header)
"""
Register C function for event finding and state interpolation.

Author: Dalton J. Moone
"""

import sys
import nrpy.c_function as cfc

def find_event_time_and_state() -> None:
    includes = ["BHaH_defines.h", "<math.h>"]
    desc = r"""@brief Finds the root of a generic event using a robust, second-order interpolation.
    Writes intersection state and affine parameter directly into global flattened SoA architecture.
    """
    name = "find_event_time_and_state"
    
    params = """PhotonStateSoA *restrict all_photons, 
                const long int num_rays, 
                const long int photon_idx, 
                const double normal[3], 
                const double dist, 
                const event_type_t event_type"""

    body = r"""
    // --- Step 0: Unpack state vectors and affine parameters from SoA ---
    double f_prev[9], f_curr[9], f_next[9];
    for (int i = 0; i < 9; i++) {
        f_prev[i] = all_photons->f_p_p[IDX_GLOBAL(i, photon_idx, num_rays)];
        f_curr[i] = all_photons->f_p[IDX_GLOBAL(i, photon_idx, num_rays)];
        f_next[i] = all_photons->f[IDX_GLOBAL(i, photon_idx, num_rays)];
    }

    const double t0 = all_photons->affine_param_p_p[photon_idx];
    const double t1 = all_photons->affine_param_p[photon_idx];
    const double t2 = all_photons->affine_param[photon_idx];

    #define PLANE_VAL(fvec) ((fvec)[1]*normal[0] + (fvec)[2]*normal[1] + (fvec)[3]*normal[2] - dist)

    double f0 = PLANE_VAL(f_prev);
    double f1 = PLANE_VAL(f_curr);
    double f2 = PLANE_VAL(f_next);

    // --- Step 1: Linear Interpolation Fallback (Robust against t=0 start) ---
    double t_linear;
    // Check if the intersection happened between t1 and t2 first (most common)
    if ( (f1 * f2 <= 0.0 || fabs(f1) < 1e-12) && fabs(f2 - f1) > 1e-15 ) { 
        t_linear = (fabs(f2 - f1) > 1e-15) ? (f2 * t1 - f1 * t2) / (f2 - f1) : t1;
    } else if ( (f0 * f1 <= 0.0 || fabs(f0) < 1e-12) && fabs(f1 - f0) > 1e-15 ) { 
        t_linear = (f1 * t0 - f0 * t1) / (f1 - f0);
    } else {
        t_linear = t1;
    }

    // --- Step 2: Quadratic Interpolation (Muller variant) ---
    const double h0 = t1 - t0;
    const double h1 = t2 - t1;
    double lambda_event = t_linear;

    if (fabs(h0) > 1e-15 && fabs(h1) > 1e-15 && fabs(h0 + h1) > 1e-15) {
        const double delta0 = (f1 - f0) / h0;
        const double delta1 = (f2 - f1) / h1;
        const double a = (delta1 - delta0) / (h1 + h0);
        const double b = a * h1 + delta1;
        const double c = f2;
        const double discriminant = b*b - 4.0*a*c;

        if (discriminant >= 0.0 && fabs(a) > 1e-16) {
            double denom = (b >= 0.0) ? (b + sqrt(discriminant)) : (b - sqrt(discriminant));
            if (fabs(denom) > 1e-16) {
                double t_quad = t2 - (2.0 * c / denom);
                // Clamp to the active interval [min(t0, t2), max(t0, t2)]
                double t_min = (t0 < t2) ? t0 : t2;
                double t_max = (t0 < t2) ? t2 : t0;
                if (t_quad >= t_min && t_quad <= t_max) {
                    lambda_event = t_quad;
                }
            }
        }
    }

    // --- Step 3: State Vector Interpolation (Lagrange) ---
    double f_event[9];
    const double t = lambda_event;

    if (fabs(h0) < 1e-15 || fabs(h1) < 1e-15) {
        double frac = (fabs(t2 - t1) > 1e-15) ? (t - t1) / (t2 - t1) : 0.0;
        for (int i = 0; i < 9; i++) f_event[i] = f_curr[i] + frac * (f_next[i] - f_curr[i]);
    } else {
        const double L0 = ((t - t1) * (t - t2)) / ((t0 - t1) * (t0 - t2));
        const double L1 = ((t - t0) * (t - t2)) / ((t1 - t0) * (t1 - t2));
        const double L2 = ((t - t0) * (t - t1)) / ((t2 - t0) * (t2 - t1));
        for (int i = 0; i < 9; i++) f_event[i] = f_prev[i] * L0 + f_curr[i] * L1 + f_next[i] * L2;
    }

    // --- Step 4: Final Output to SoA ---
    if (event_type == SOURCE_EVENT) {
        all_photons->source_event_lambda[photon_idx] = lambda_event;
        all_photons->source_event_found[photon_idx] = true;
        for (int i = 0; i < 9; i++) {
            all_photons->source_event_f_intersect[IDX_GLOBAL(i, photon_idx, num_rays)] = f_event[i];
        }
    } else if (event_type == WINDOW_EVENT) {
        all_photons->window_event_lambda[photon_idx] = lambda_event;
        all_photons->window_event_found[photon_idx] = true;
        for (int i = 0; i < 9; i++) {
            all_photons->window_event_f_intersect[IDX_GLOBAL(i, photon_idx, num_rays)] = f_event[i];
        }
    }
    #undef PLANE_VAL
    """

    cfc.register_CFunction(
        includes=includes, desc=desc, name=name, params=params, body=body
    )
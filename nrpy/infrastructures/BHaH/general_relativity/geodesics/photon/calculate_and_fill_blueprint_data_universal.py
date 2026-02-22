"""
Generates the C "finalizer" engine to process a completed ray trace.
Refactored to read from a flattened SoA master struct for the final 
physical results extraction.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc

def calculate_and_fill_blueprint_data_universal() -> None:
    """
    Generate and register the C finalization engine using SoA data access.
    """
    name = "calculate_and_fill_blueprint_data_universal"
    cfunc_type = "blueprint_data_t"
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = "@brief Processes a photon's final state via SoA to compute blueprint quantities."

    params = """const commondata_struct *restrict commondata, 
                const PhotonStateSoA *restrict all_photons, 
                const long int num_rays, 
                const long int photon_idx,
                const double window_center[3], 
                const double n_x[3], 
                const double n_y[3]"""

    body = r"""
    blueprint_data_t result = {0};
    result.termination_type = all_photons->status[photon_idx];

    // --- Window Data Population ---
    if (all_photons->window_event_found[photon_idx]) {
        const double pos_w_cart[3] = {
            all_photons->window_event_f_intersect[IDX_GLOBAL(1, photon_idx, num_rays)],
            all_photons->window_event_f_intersect[IDX_GLOBAL(2, photon_idx, num_rays)],
            all_photons->window_event_f_intersect[IDX_GLOBAL(3, photon_idx, num_rays)]
        };
        const double vec_w[3] = {pos_w_cart[0]-window_center[0], pos_w_cart[1]-window_center[1], pos_w_cart[2]-window_center[2]};
        
        result.y_w = vec_w[0]*n_x[0] + vec_w[1]*n_x[1] + vec_w[2]*n_x[2];
        result.z_w = vec_w[0]*n_y[0] + vec_w[1]*n_y[1] + vec_w[2]*n_y[2];
        result.L_w = all_photons->window_event_f_intersect[IDX_GLOBAL(8, photon_idx, num_rays)];
    }

    // --- Termination Dispatch ---
    if (all_photons->status[photon_idx] == TERMINATION_TYPE_SOURCE_PLANE) {
        // Updated to pass SoA pointers
        handle_source_plane_intersection(all_photons, num_rays, photon_idx, commondata, &result);

    } else if (all_photons->status[photon_idx] == TERMINATION_TYPE_CELESTIAL_SPHERE) {
        const double x = all_photons->f[IDX_GLOBAL(1, photon_idx, num_rays)];
        const double y = all_photons->f[IDX_GLOBAL(2, photon_idx, num_rays)];
        const double z = all_photons->f[IDX_GLOBAL(3, photon_idx, num_rays)];
        const double r = sqrt(SQR(x) + SQR(y) + SQR(z));
        if (r > 1e-9) {
            result.final_theta = acos(z / r);
            result.final_phi = atan2(y, x);
        }
    }

    return result;
    """

    cfc.register_CFunction(includes=includes, desc=desc, cfunc_type=cfunc_type, name=name, params=params, body=body)

"""
Generates the C orchestrator for geometric event detection.
Refactored to eradicate AoS event structs and monitor trajectories directly
via flattened PhotonStateSoA global arrays.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc
import nrpy.infrastructures.BHaH.BHaH_defines_h as Bdefines_h

# Define the C-code for event types, geometry, and helper functions
event_mgmt_c_code = r"""
    // ==========================================
    // Event Detection Types and Geometry
    // ==========================================
    
    // Define the function pointer type for event functions
    typedef double (*event_function_t)(const double *f, long int num_rays, long int photon_idx, const void *params);

    typedef struct {
        double normal[3];
        double d;
    } plane_event_params;

    /**
     * @brief Computes the signed distance of a photon from a defined plane.
     * Uses the 1D flattened SoA mapping: ((component) * (num_rays) + (ray_id)).
     */
    static inline double plane_event_func(const double *f, long int num_rays, long int photon_idx, const void *params) {
        const plane_event_params *p = (const plane_event_params *)params;
        // Unpack coordinates from the flat SoA layout
        const double x = f[1 * num_rays + photon_idx]; 
        const double y = f[2 * num_rays + photon_idx]; 
        const double z = f[3 * num_rays + photon_idx]; 
        
        return p->normal[0] * x + p->normal[1] * y + p->normal[2] * z - p->d;
    }
"""

# Register the defines so they appear in BHaH_defines.h
Bdefines_h.register_BHaH_defines("photon_03_event_management", event_mgmt_c_code)

def event_detection_manager() -> None:
    """
    Generate and register the C event detection manager.
    Logic is performed directly on the global SoA pointers for SIMT efficiency.
    """
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "<math.h>"]
    desc = "@brief Detects crossings of the window and source planes using SoA data."
    name = "event_detection_manager"

    # Signature updated to use the master SoA struct and global indexing
    params = """
        PhotonStateSoA *restrict all_photons, 
        const long int num_rays, 
        const long int photon_idx, 
        const commondata_struct *restrict commondata
        """

    body = r"""
    // Helper lambda-like macro for local plane evaluation within the SoA context
    #define EVAL_PLANE(normal, dist, state_ptr, r_idx, total_rays) ( \
        state_ptr[IDX_GLOBAL(1, r_idx, total_rays)]*(normal)[0] + \
        state_ptr[IDX_GLOBAL(2, r_idx, total_rays)]*(normal)[1] + \
        state_ptr[IDX_GLOBAL(3, r_idx, total_rays)]*(normal)[2] - (dist) )

    // --- Window Plane Detection ---
    if (!all_photons->window_event_found[photon_idx]) {
        double w_normal[3] = {
            commondata->window_center_x - commondata->camera_pos_x,
            commondata->window_center_y - commondata->camera_pos_y,
            commondata->window_center_z - commondata->camera_pos_z
        };
        double mag_inv = 1.0 / sqrt(SQR(w_normal[0]) + SQR(w_normal[1]) + SQR(w_normal[2]));
        for(int i=0; i<3; i++) w_normal[i] *= mag_inv;
        double w_dist = commondata->window_center_x*w_normal[0] + commondata->window_center_y*w_normal[1] + commondata->window_center_z*w_normal[2];

        // FIXED: Use an epsilon to prevent false crossings for photons starting EXACTLY on the plane
        double w_val = EVAL_PLANE(w_normal, w_dist, all_photons->f, photon_idx, num_rays);
        bool on_pos_curr = (w_val > 1e-10); 
        
        if (on_pos_curr != all_photons->on_positive_side_of_window_prev[photon_idx]) {
             // Crossing found: Root-find and write directly to all_photons->window_event_f_intersect
             find_event_time_and_state(all_photons, num_rays, photon_idx, w_normal, w_dist, WINDOW_EVENT);
        }
        all_photons->on_positive_side_of_window_prev[photon_idx] = on_pos_curr;
    }

    // --- Source Plane Detection ---
    if (!all_photons->source_event_found[photon_idx]) {
        double s_normal[3] = {commondata->source_plane_normal_x, commondata->source_plane_normal_y, commondata->source_plane_normal_z};
        double s_dist = commondata->source_plane_center_x*s_normal[0] + commondata->source_plane_center_y*s_normal[1] + commondata->source_plane_center_z*s_normal[2];

        // FIXED: Added epsilon check for source plane as well
        double s_val = EVAL_PLANE(s_normal, s_dist, all_photons->f, photon_idx, num_rays);
        bool on_pos_curr = (s_val > 1e-10);
        
        if (on_pos_curr != all_photons->on_positive_side_of_source_prev[photon_idx]) {
             find_event_time_and_state(all_photons, num_rays, photon_idx, s_normal, s_dist, SOURCE_EVENT);
        }
        all_photons->on_positive_side_of_source_prev[photon_idx] = on_pos_curr;
    }
    #undef EVAL_PLANE
    """

    cfc.register_CFunction(includes=includes, desc=desc, name=name, params=params, body=body)
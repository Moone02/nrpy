"""
Generates the C engine to handle a fallback source plane intersection.

This module registers the C function responsible for processing photons that
intersect the source plane. It calculates local coordinates on the plane,
checks if the photon hit the active "glowing" region, and records the data.

Author: Dalton J. Moone
"""

import sys
import os

# Import NRPy core modules
import nrpy.c_function as cfc
import nrpy.infrastructures.BHaH.BHaH_defines_h as Bdefines_h


def handle_source_plane_intersection() -> None:
    """
    Generate and register the C engine for processing source plane intersections.

    This function generates the C code that is called when a photon hits the
    fallback source plane. It transforms the 3D intersection point into local
    2D texture coordinates (y_s, z_s) using a Gram-Schmidt-like process to
    create an orthonormal basis from the plane's normal and "up" vectors.

    It then checks if the hit is within the active annulus (r_min < r < r_max)
    and populates the 'final_blueprint_data' struct if valid.
    """
    # Step 1: Define specific C headers and function metadata
    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
        "<math.h>",
        "<stdbool.h>",
    ]
    
    desc = r"""@brief Handles a source plane intersection by checking bounds and populating the blueprint.
    
    This function performs a coordinate transformation from the global 3D Cartesian basis 
    to a local 2D basis on the source plane.

    @param[in]  source_plane_event  Pointer to the event data struct containing the intersection point.
    @param[in]  commondata          Pointer to commondata struct with runtime parameters (plane definition).
    @param[out] final_blueprint_data Pointer to the final output struct to be populated if valid.
    @return True if the intersection is valid (within the active annulus) and processed; false otherwise.
    """
    
    name = "handle_source_plane_intersection"
    cfunc_type = "bool"
    params = """
    const event_data_struct *restrict source_plane_event,
    const commondata_struct *restrict commondata,
    blueprint_data_t *restrict final_blueprint_data
    """

    # Step 2: Define the C body with improved readability and comments
    # We use raw strings (r"...") to avoid escaping backslashes.
    body = r"""
    // ---------------------------------------------------------
    // 1. Unpack Configuration and Intersection Data
    // ---------------------------------------------------------
    const double intersection_pos[3] = {
        source_plane_event->f_event[1], 
        source_plane_event->f_event[2], 
        source_plane_event->f_event[3]
    };
    
    const double source_plane_center[3] = {
        commondata->source_plane_center_x, 
        commondata->source_plane_center_y, 
        commondata->source_plane_center_z
    };
    
    const double source_plane_normal[3] = {
        commondata->source_plane_normal_x, 
        commondata->source_plane_normal_y, 
        commondata->source_plane_normal_z
    };
    
    const double source_up_vector[3] = {
        commondata->source_up_vec_x, 
        commondata->source_up_vec_y, 
        commondata->source_up_vec_z
    };

    // ---------------------------------------------------------
    // 2. Construct Orthonormal Basis (s_x, s_y, s_z)
    // ---------------------------------------------------------
    // s_z is simply the plane normal.
    double s_z[3] = { source_plane_normal[0], source_plane_normal[1], source_plane_normal[2] };

    // Calculate s_x = Up x s_z (Cross Product).
    // This creates a vector orthogonal to the normal, roughly aligned with "right".
    double s_x[3];
    s_x[0] = source_up_vector[1]*s_z[2] - source_up_vector[2]*s_z[1];
    s_x[1] = source_up_vector[2]*s_z[0] - source_up_vector[0]*s_z[2];
    s_x[2] = source_up_vector[0]*s_z[1] - source_up_vector[1]*s_z[0];
    
    double mag_s_x = sqrt(s_x[0]*s_x[0] + s_x[1]*s_x[1] + s_x[2]*s_x[2]);

    // Safety: Handle the edge case where the Up vector is parallel to the Normal.
    // If parallel, the cross product is zero. We pick an arbitrary axis to recover.
    if (mag_s_x < 1e-9) {
        double alternative_up[3] = {1.0, 0.0, 0.0};
        
        // If normal is also X-aligned, switch alternative to Y.
        if (fabs(s_z[0]) > 0.999) {
            alternative_up[0] = 0.0; 
            alternative_up[1] = 1.0;
        }
        
        // Recompute s_x with the alternative up vector
        s_x[0] = alternative_up[1]*s_z[2] - alternative_up[2]*s_z[1];
        s_x[1] = alternative_up[2]*s_z[0] - alternative_up[0]*s_z[2];
        s_x[2] = alternative_up[0]*s_z[1] - alternative_up[1]*s_z[0];
        
        mag_s_x = sqrt(s_x[0]*s_x[0] + s_x[1]*s_x[1] + s_x[2]*s_x[2]);
    }

    // Normalize s_x
    double inv_mag_s_x = 1.0 / mag_s_x;
    s_x[0] *= inv_mag_s_x;
    s_x[1] *= inv_mag_s_x;
    s_x[2] *= inv_mag_s_x;

    // Calculate s_y = s_z x s_x (Cross Product).
    // Since s_z and s_x are orthonormal, s_y is automatically normalized.
    double s_y[3];
    s_y[0] = s_z[1]*s_x[2] - s_z[2]*s_x[1];
    s_y[1] = s_z[2]*s_x[0] - s_z[0]*s_x[2];
    s_y[2] = s_z[0]*s_x[1] - s_z[1]*s_x[0];

    // ---------------------------------------------------------
    // 3. Project Intersection onto Local Plane Coordinates
    // ---------------------------------------------------------
    // Vector from plane center to intersection point
    const double vec_s[3] = {
        intersection_pos[0] - source_plane_center[0],
        intersection_pos[1] - source_plane_center[1],
        intersection_pos[2] - source_plane_center[2]
    };

    // Project vec_s onto the basis vectors (Dot Products)
    const double y_s = vec_s[0]*s_x[0] + vec_s[1]*s_x[1] + vec_s[2]*s_x[2];
    const double z_s = vec_s[0]*s_y[0] + vec_s[1]*s_y[1] + vec_s[2]*s_y[2];

    // ---------------------------------------------------------
    // 4. Validate Bounds (Annulus Check)
    // ---------------------------------------------------------
    const double r_s_sq = y_s*y_s + z_s*z_s;
    const double r_min_sq = commondata->source_r_min * commondata->source_r_min;
    const double r_max_sq = commondata->source_r_max * commondata->source_r_max;

    if (r_s_sq >= r_min_sq && r_s_sq <= r_max_sq) {
        // Valid Hit: Populate the blueprint data
        final_blueprint_data->termination_type = TERMINATION_TYPE_SOURCE_PLANE;
        final_blueprint_data->y_s = y_s;
        final_blueprint_data->z_s = z_s;
        final_blueprint_data->t_s = source_plane_event->t_event;
        
        // f_event[8] typically holds the affine parameter or a conserved quantity 
        // depending on integrator implementation, here mapped to L_s.
        final_blueprint_data->L_s = source_plane_event->f_event[8]; 
        
        return true;
    }

    // Intersection occurred, but it was in the transparent hole or beyond outer rim.
    return false;
    """

    # Step 3: Register the C function
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        name=name,
        cfunc_type=cfunc_type,
        params=params,
        body=body,
    )


if __name__ == "__main__":
    import logging
    import nrpy.params as par

    # Ensure local modules can be imported
    sys.path.append(os.getcwd())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestHandleSourcePlaneIntersection")

    logger.info("Test: Generating C-code for Source Plane Intersection Handler...")

    try:        
        # 1. Run the Generator
        logger.info(" -> Calling handle_source_plane_intersection()...")
        handle_source_plane_intersection()

        # 2. Validation
        cfunc_name = "handle_source_plane_intersection"

        # Check C Function Registration
        if cfunc_name not in cfc.CFunction_dict:
            raise RuntimeError(
                f"FAIL: '{cfunc_name}' was not registered in cfc.CFunction_dict."
            )

        logger.info(" -> PASS: '%s' function registered successfully.", cfunc_name)

        # 3. Output Files to Current Directory for Inspection
        logger.info(" -> Writing generated code to disk...")
        
        # We also attempt to output BHaH_defines.h if needed, but primarily the C file.
        Bdefines_h.output_BHaH_defines_h(project_dir=".")
        
        for func_name, c_function in cfc.CFunction_dict.items():
            filename = f"{func_name}.c"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(c_function.full_function)
            logger.info("    ... Wrote %s", filename)

        logger.info(" -> Success! All files generated.")

    except Exception as e:
        logger.error(" -> FAIL: handle_source_plane_intersection test failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
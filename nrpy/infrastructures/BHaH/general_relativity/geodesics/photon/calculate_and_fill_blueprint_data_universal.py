"""
Generates the C "finalizer" engine to process a completed ray trace.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc


def calculate_and_fill_blueprint_data_universal() -> None:
    """
    Generate and register the C finalization engine.

    This function generates the C code that is called once for each photon after
    its integration is complete. It acts as a dispatcher based on the photon's
    final termination status, calling the appropriate helper engines to compute
    the final physical and geometric quantities that are saved to the output file.
    """
    # -------------------------------------------------------------------------
    # Step 1: Define C Function Signature and Metadata
    # -------------------------------------------------------------------------
    name = "calculate_and_fill_blueprint_data_universal"
    cfunc_type = "blueprint_data_t"

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    
    desc = r"""@brief Processes a photon's final state to compute all blueprint quantities.

    This finalizer engine is called once for each completed ray. It dispatches
    to the appropriate helper engine based on the photon's termination status
    to calculate the final physical results (e.g., intensity, texture coordinates)
    and populates the `blueprint_data_t` struct for output.

    @param[in]  commondata     Pointer to commondata struct with runtime parameters.
    @param[in]  photon         Pointer to the final state of the completed photon.
    @param[in]  window_center  3D Cartesian coordinates of the window plane's center.
    @param[in]  n_x, n_y       Orthonormal basis vectors for the window plane.
    @return A fully populated `blueprint_data_t` struct ready for output.
    """
    
    params = """const commondata_struct *restrict commondata, 
                const PhotonState *restrict photon,
                const double window_center[3], 
                const double n_x[3], 
                const double n_y[3]"""

    # -------------------------------------------------------------------------
    # Step 2: Define the C Function Body
    # -------------------------------------------------------------------------
    # Note: TERMINATION_TYPE_DISK logic has been removed as it is deprecated.
    body = r"""
    // Initialize all fields to zero.
    blueprint_data_t result = {0};
    result.termination_type = photon->status;

    // --- Step 1: Always populate window data if a crossing was found ---
    if (photon->window_event_data.found) {
        const double *f_event = photon->window_event_data.f_event;
        const double pos_w_cart[3] = {f_event[1], f_event[2], f_event[3]};
        const double vec_w[3] = {
            pos_w_cart[0] - window_center[0], 
            pos_w_cart[1] - window_center[1], 
            pos_w_cart[2] - window_center[2]
        };
        
        // Project intersection point onto window's orthonormal basis to get texture coordinates.
        result.y_w = vec_w[0]*n_x[0] + vec_w[1]*n_x[1] + vec_w[2]*n_x[2];
        result.z_w = vec_w[0]*n_y[0] + vec_w[1]*n_y[1] + vec_w[2]*n_y[2];
        
        result.L_w = f_event[8];
        result.t_w = photon->window_event_data.t_event;
    }

    // --- Step 2: Populate remaining fields based on the specific termination type ---
    if (photon->status == TERMINATION_TYPE_SOURCE_PLANE) {
        // Call the source plane helper to validate the hit and calculate geometric properties.
        handle_source_plane_intersection(&photon->source_event_data, commondata, &result);

    } else if (photon->status == TERMINATION_TYPE_CELESTIAL_SPHERE) {
        // Convert the final Cartesian position to spherical polar angles.
        const double *final_f = photon->f;
        const double x = final_f[1];
        const double y = final_f[2];
        const double z = final_f[3];
        const double r = sqrt(SQR(x) + SQR(y) + SQR(z));
        
        if (r > 1e-9) {
            result.final_theta = acos(z / r);
            result.final_phi = atan2(y, x);
        }
    }
    // For FAILURE types, no other fields need to be set.

    return result;
    """

    # -------------------------------------------------------------------------
    # Step 3: Register the C Function
    # -------------------------------------------------------------------------
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
        include_CodeParameters_h= False,
    )


if __name__ == "__main__":
    import logging
    import os
    import sys

    # Ensure local modules can be imported
    sys.path.append(os.getcwd())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestBlueprintData")

    logger.info("Test: Generating Blueprint Data C-code...")

    try:
        # 1. Run the Generator
        calculate_and_fill_blueprint_data_universal()

        # 2. Validation
        cfunc_name = "calculate_and_fill_blueprint_data_universal"

        # Check Registration
        if cfunc_name not in cfc.CFunction_dict:
            raise RuntimeError(f"FAIL: '{cfunc_name}' was not registered.")

        cfunc = cfc.CFunction_dict[cfunc_name]
        logger.info(" -> PASS: '%s' registered.", cfunc_name)

        # 3. Output to file
        filename = f"{cfunc_name}.c"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(cfunc.full_function)
        logger.info("    ... Wrote %s", filename)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(" -> FAIL: Test failed with error: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)
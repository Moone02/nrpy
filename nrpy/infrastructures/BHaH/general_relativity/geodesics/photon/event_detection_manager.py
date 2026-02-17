"""
Generates the C orchestrator for geometric event detection.

This module registers the 'event_detection_manager' C function, which is responsible
for monitoring the photon's trajectory at every integration step. It detects if
the photon has crossed specific geometric boundaries (the camera window or the
source plane) and triggers the high-precision interpolation engine if a crossing occurs.

Author: Dalton J. Moone
"""

import sys
import os

import nrpy.c_function as cfc


def event_detection_manager() -> None:
    """
    Generate and register the C event detection manager.

    This function generates the C code that serves as the "step-watcher" during
    integration. It performs the following logic:
    1. Checks if the photon has already been flagged as 'found' for an event.
    2. If not, it calculates the signed distance of the photon to the Window and Source planes.
    3. Checks for a sign change in distance between the current and previous steps.
    4. If a sign change (crossing) is detected, it invokes `find_event_time_and_state()`
       to solve for the exact intersection time.
    """
    # -------------------------------------------------------------------------
    # Step 1: Define C Function Metadata
    # -------------------------------------------------------------------------
    # Defines and prototypes required for the C function to compile.
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "<math.h>"]
    
    desc = r"""@brief Detects crossings of the window and source planes.

    This orchestrator is called at each integration step. It determines if a
    sign change has occurred in the photon's distance to either the window or
    source plane. If so, it calls the `find_event_time_and_state()` engine to
    accurately interpolate the intersection point.

    @param[in]      f_prev, f_curr, f_next  State vectors at three consecutive steps.
    @param[in]      lambda_prev, lambda_curr, lambda_next Affine parameters for the three steps.
    @param[in]      commondata              Pointer to commondata struct with runtime parameters.
    @param[in,out]  on_positive_side_of_window_prev Pointer to the state of the photon relative to the window plane at the previous step.
    @param[in,out]  on_positive_side_of_source_prev Pointer to the state of the photon relative to the source plane at the previous step.
    @param[out]     window_event            Pointer to the event_data_struct for window crossings.
    @param[out]     source_plane_event      Pointer to the event_data_struct for source plane crossings.
    """
    
    name = "event_detection_manager"
    
    params = """
        const double f_prev[9], const double f_curr[9], const double f_next[9],
        double lambda_prev, double lambda_curr, double lambda_next,
        const commondata_struct *restrict commondata,
        bool *on_positive_side_of_window_prev,
        bool *on_positive_side_of_source_prev,
        event_data_struct *restrict window_event,
        event_data_struct *restrict source_plane_event
        """

    # -------------------------------------------------------------------------
    # Step 2: Define C Function Body
    # -------------------------------------------------------------------------
    # The body checks plane equations: n_i * x^i = d.
    # If (n * x_prev - d) and (n * x_curr - d) have different signs, a crossing occurred.
    body = r"""
    // ------------------------------------------------------------------------
    // Logic Block 1: Window Plane Detection
    // ------------------------------------------------------------------------
    // Only proceed if we haven't already found the window event for this photon.
    if (!window_event->found) {
        
        // 1.a. Calculate the Window Plane Normal Vector
        //      Normal = (Window Center) - (Camera Position)
        double window_plane_normal[3] = {
            commondata->window_center_x - commondata->camera_pos_x,
            commondata->window_center_y - commondata->camera_pos_y,
            commondata->window_center_z - commondata->camera_pos_z
        };

        // 1.b. Normalize the vector
        const double mag_w_norm_sq = SQR(window_plane_normal[0]) + 
                                     SQR(window_plane_normal[1]) + 
                                     SQR(window_plane_normal[2]);
        const double mag_w_norm = sqrt(mag_w_norm_sq);

        if (mag_w_norm > 1e-12) {
            const double inv_mag_w_norm = 1.0 / mag_w_norm;
            for(int i=0; i<3; i++) window_plane_normal[i] *= inv_mag_w_norm;
        }

        // 1.c. Calculate Plane Distance 'd' from origin
        //      d = Normal . (Window Center)
        const double window_plane_dist = commondata->window_center_x * window_plane_normal[0] +
                                         commondata->window_center_y * window_plane_normal[1] +
                                         commondata->window_center_z * window_plane_normal[2];

        // 1.d. Prepare parameters for the distance function
        plane_event_params window_params = {
            {window_plane_normal[0], window_plane_normal[1], window_plane_normal[2]}, 
            window_plane_dist
        };

        // 1.e. Check for Crossing
        //      Evaluate which side of the plane the photon is on at the 'next' step.
        //      If it differs from the 'prev' step, a crossing occurred in the interval.
        bool on_positive_side_curr = (plane_event_func(f_next, &window_params) > 0);

        if (on_positive_side_curr != *on_positive_side_of_window_prev) {
            // A crossing occurred!
            // Call the high-accuracy interpolator to find the exact time 'lambda' where dist == 0.
            find_event_time_and_state(f_prev, f_curr, f_next, 
                                      lambda_prev, lambda_curr, lambda_next,
                                      plane_event_func, &window_params, window_event);
        }

        // 1.f. Update state for the next integration step
        *on_positive_side_of_window_prev = on_positive_side_curr;
    }

    // ------------------------------------------------------------------------
    // Logic Block 2: Source Plane Detection
    // ------------------------------------------------------------------------
    // Only proceed if we haven't already found the source event.
    if (!source_plane_event->found) {
        
        // 2.a. Retrieve Source Plane Geometry (pre-computed in commondata)
        const double source_plane_normal[3] = {
            commondata->source_plane_normal_x,
            commondata->source_plane_normal_y,
            commondata->source_plane_normal_z
        };

        // 2.b. Calculate Plane Distance 'd'
        const double source_plane_dist = commondata->source_plane_center_x * source_plane_normal[0] +
                                         commondata->source_plane_center_y * source_plane_normal[1] +
                                         commondata->source_plane_center_z * source_plane_normal[2];

        // 2.c. Prepare parameters
        plane_event_params source_params = {
            {source_plane_normal[0], source_plane_normal[1], source_plane_normal[2]}, 
            source_plane_dist
        };

        // 2.d. Check for Crossing
        bool on_positive_side_curr = (plane_event_func(f_next, &source_params) > 0);

        if (on_positive_side_curr != *on_positive_side_of_source_prev) {
            // A crossing occurred. Invoke the interpolator.
            find_event_time_and_state(f_prev, f_curr, f_next, 
                                      lambda_prev, lambda_curr, lambda_next,
                                      plane_event_func, &source_params, source_plane_event);
        }

        // 2.e. Update state
        *on_positive_side_of_source_prev = on_positive_side_curr;
    }
    """

    # -------------------------------------------------------------------------
    # Step 3: Register with CFunction Dictionary
    # -------------------------------------------------------------------------
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        name=name,
        params=params,
        body=body,
    )


if __name__ == "__main__":
    import logging

    # Configure logging to print only the message content
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    logger = logging.getLogger("TestEventDetectionManager")

    logger.info("Test: Generating Event Detection Manager C-code...")

    try:
        # 1. Run the Generator
        logger.info(" -> Calling event_detection_manager()...")
        event_detection_manager()

        # 2. Validation
        cfunc_name = "event_detection_manager"

        # Check C Function Registration
        if cfunc_name not in cfc.CFunction_dict:
            raise RuntimeError(
                f"FAIL: '{cfunc_name}' was not registered in cfc.CFunction_dict."
            )

        logger.info(" -> PASS: '%s' function registered successfully.", cfunc_name)

        # 3. Output Files to Current Directory for Inspection
        output_filename = f"{cfunc_name}.c"
        c_function = cfc.CFunction_dict[cfunc_name]
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(c_function.full_function)
        
        logger.info("    ... Wrote %s for inspection.", output_filename)
        logger.info(" -> Success! Test complete.")

    except Exception as e:
        logger.error(" -> FAIL: event_detection_manager test failed.")
        logger.exception(e)
        sys.exit(1)
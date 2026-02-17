"""
Register C function for setting the initial state vector for a photon.

This module registers the 'set_initial_conditions_cartesian_{spacetime_name}' C function.
It generates the C code that sets the complete 9-component initial state vector (f[9])
for a single light ray. It orchestrates a sequence of geometric calculations and calls
other C engines (g4DD_metric and p0_reverse) to accomplish this.

Author: Dalton J. Moone 
"""

# Step 0.a: Import standard Python modules
import logging
import sys

# Step 0.c: Import NRPy core modules
import nrpy.c_function as cfc




def set_initial_conditions_cartesian(spacetime_name: str) -> None:
    """
    Generate and register the C orchestrator for setting photon initial conditions.

    :param spacetime_name: Name of the spacetime (used to call the specific metric function).
    """
    # Step 1: Define C function metadata
    # We include math.h for sqrt(), stdio.h/stdlib.h for error handling.
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "math.h", "stdio.h", "stdlib.h"]
    
    desc = f"""@brief Sets the full initial state for a light ray in Cartesian coordinates.

    This function orchestrates the setup of the initial state vector f[9] for a
    single photon in the {spacetime_name} spacetime.

    It performs the following steps:
    1. Sets initial position (f[0]..f[3]) using 'commondata->t_start' and 'commondata->camera_pos'.
    2. Computes initial spatial momentum (f[5]..f[7]) from the aiming vector.
    3. Calls 'g4DD_metric_{spacetime_name}' to get metric components.
    4. Calls 'p0_reverse' to solve the null condition for time momentum (f[4]).
    5. Initializes path length (f[8]) to zero.

    Input:
        commondata: Runtime parameters (contains t_start and camera_pos).
        target_pos: 3D Cartesian coordinates of the target pixel.
    Output:
        f[9]: The 9-component output array for the initial state vector."""

    name = f"set_initial_conditions_cartesian_{spacetime_name}"
    

    params = (
        "const commondata_struct *restrict commondata, "
        "const double target_pos[3], "
        "double f[9]"
    )

    # Step 2: Generate C body
    body = f"""
    // --- Step 1: Set the initial position to the camera's location ---
    // We access t_start and camera_pos directly from the commondata struct.
    f[0] = commondata->t_start;       // t
    f[1] = commondata->camera_pos_x; // x
    f[2] = commondata->camera_pos_y; // y
    f[3] = commondata->camera_pos_z; // z

    // --- Step 2: Calculate the aiming vector V and set spatial momentum ---
    // The initial spatial momentum p^i (f[5], f[6], f[7]) is parallel to the aiming vector.
    // Note: We use commondata->camera_pos here as well.
    const double V_x = target_pos[0] - commondata->camera_pos_x;
    const double V_y = target_pos[1] - commondata->camera_pos_y;
    const double V_z = target_pos[2] - commondata->camera_pos_z;
    const double mag_V = sqrt(V_x*V_x + V_y*V_y + V_z*V_z);

    if (mag_V > 1e-12) {{
        // Normalize to improve numerical stability and precision.
        const double inv_mag_V = 1.0 / mag_V;
        f[5] = V_x * inv_mag_V; // p^x
        f[6] = V_y * inv_mag_V; // p^y
        f[7] = V_z * inv_mag_V; // p^z
    }} else {{
        fprintf(stderr, "ERROR in set_initial_conditions_cartesian_{spacetime_name}: Camera position exactly matches target position.\\n");
        fprintf(stderr, "       Cannot determine photon aiming direction (magnitude of vector is near-zero). Aborting.\\n");
        exit(1);
    }}

    // --- Step 3: Calculate the time component p^0 using the null condition ---
    // We allocate the metric struct on the STACK.
    metric_struct metric;

    // 3.a: Compute metric components at current position f[0]..f[3]
    g4DD_metric_{spacetime_name}(commondata, f, &metric);

    // 3.b: Solve quadratic constraint for p^0 (f[4])
    // p0_reverse reads spatial momentum from f[5]..f[7] and writes result to &f[4]
    p0_reverse(&metric, f, &f[4]);

    // --- Step 4: Initialize integrated path length ---
    f[8] = 0.0;
    """

    print(f" -> Generating C worker function: {name} (Spacetime: {spacetime_name})...")

    # Step 3: Register the C function
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    print(f"    ... {name}() registration complete.")

    

if __name__ == "__main__":
    import os

    # Ensure local modules can be imported
    sys.path.append(os.getcwd())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestSetInitialConditions")

    SPACETIME = "KerrSchild_Cartesian"
    logger.info("Test: Generating Initial Conditions C-code for %s...", SPACETIME)

    try:
        # 1. Run the Generator
        logger.info(" -> Calling set_initial_conditions_cartesian()...")
        set_initial_conditions_cartesian(SPACETIME)

        # 2. Validation
        cfunc_name = f"set_initial_conditions_cartesian_{SPACETIME}"
        
        if cfunc_name not in cfc.CFunction_dict:
            raise RuntimeError(
                f"FAIL: '{cfunc_name}' was not registered in cfc.CFunction_dict."
            )
        logger.info(" -> PASS: '%s' function registered successfully.", cfunc_name)

        # 3. Output Files
        filename = f"{cfunc_name}.c"
        cfunc = cfc.CFunction_dict[cfunc_name]
        with open(filename, "w", encoding="utf-8") as file:
            file.write(cfunc.full_function)
        logger.info(" -> Written to %s", filename)

    except Exception as e:
        logger.error(" -> FAIL: set_initial_conditions test failed with error: %s", e)
        sys.exit(1)
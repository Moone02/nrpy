"""
Generates the main() C function for the photon geodesic integrator project.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc


def main(spacetime_name: str) -> None:
    """
    Register the master orchestrator C function.

    This function generates the C code for the main() function. It acts as the
    master orchestrator, managing the simulation's lifecycle.

    :param spacetime_name: The name of the spacetime (e.g., "KerrSchild_Cartesian").
    """
    # Per project standards, define local variables for all register_CFunction args.
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "stdio.h", "stdlib.h"]
    desc = f"""@brief Main entry point for the geodesic integrator.

    Orchestrates the simulation for the {spacetime_name} spacetime.
    """
    cfunc_type = "int"
    name = "main"
    params = "int argc, const char *argv[]"

    # The body is algorithmic, not symbolic, so it is defined as a raw C string.
    # We use Python f-strings to inject the spacetime_name into the C code.
    body = f"""
    // --- Step 1: Initialize Core Data Structures ---
    commondata_struct commondata;

    // --- Step 2: Set Default Parameters and Parse User Input ---
    commondata_struct_set_to_default(&commondata);
    cmdline_input_and_parfile_parser(&commondata, argc, argv);


    // --- Step 3: Print Simulation Banner ---
    printf("=============================================\\n");
    printf("  Photon Geodesic Integrator (Batch Mode)  \\n");
    printf("=============================================\\n");
    printf("Spacetime: {spacetime_name}\\n");
    printf("Spin (a): %.3f\\n", commondata.a_spin);
    printf("Scan Resolution: %d x %d\\n", commondata.scan_density, commondata.scan_density);

    // --- Step 4: Main Logic Dispatcher ---
    long int num_rays = (long int)commondata.scan_density * commondata.scan_density;
    blueprint_data_t *results_buffer = (blueprint_data_t *)malloc(sizeof(blueprint_data_t) * num_rays);
    if (results_buffer == NULL) {{ 
        fprintf(stderr, "Error: Failed to allocate results buffer.\\n");
        exit(1); 
    }}

    // Call the Numerical pipeline orchestrator.
    batch_integrator_numerical(&commondata, num_rays, results_buffer);

    printf("Scan finished. Writing %ld ray results to light_blueprint.bin...\\n", num_rays);
    FILE *fp_blueprint = fopen("light_blueprint.bin", "wb");
    if (fp_blueprint == NULL) {{ perror("Error opening blueprint file"); exit(1); }}
    fwrite(results_buffer, sizeof(blueprint_data_t), num_rays, fp_blueprint);
    fclose(fp_blueprint);
    free(results_buffer);

    // --- Step 6: Cleanup ---
    printf("\\nRun complete.\\n");
    return 0;
    """

    # Register the C function.
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
    )


if __name__ == "__main__":
    import logging
    import os
    import sys

    # Ensure local directory is in path for imports if needed
    sys.path.append(os.getcwd())

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GenerateMain")

    # Define a default spacetime for testing the generator
    TEST_SPACETIME = "KerrSchild_Cartesian"
    logger.info("Test: Generating main() C-code for spacetime: %s...", TEST_SPACETIME)

    try:
        main(TEST_SPACETIME)
        func_name = "main"
        if func_name in cfc.CFunction_dict:
            filename = f"{func_name}.c"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(cfc.CFunction_dict[func_name].full_function)
            logger.info(" -> Success! Wrote %s", filename)
        else:
            raise RuntimeError(f"Function {func_name} not registered.")
    except (RuntimeError, OSError) as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)

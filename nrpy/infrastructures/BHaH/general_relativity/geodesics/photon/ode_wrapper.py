"""
Register C dispatcher function for the internal RKF45 ODE solver.

This module registers the 'ode_wrapper_{spacetime}' C function.
It computes the metric and Christoffel symbols before invoking the
RHS calculation engine, specifically for use with the internal RKF45 kernel.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc


def ode_wrapper(spacetime_name: str) -> None:
    """
    Generate the C dispatcher function for the internal RKF45 solver.

    This function calls the project-specific geometry engines and
    the RHS engine, providing a clean interface for the RKF45 stepper.

    :param spacetime_name: String used to define metric in analytic_spacetimes.py.
    """
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = f"""@brief Internal dispatcher for photon geodesics in {spacetime_name}.
        
        Computes the local metric and connections, then calls the RHS calculation routine.
        
        Input:
            y[9]: Current state vector.
            commondata: Pointer to the simulation's common data parameters.
        Output:
            f[9]: Computed derivatives (RHS)."""

    # Changed type to void and removed GSL-specific void* params
    cfunc_type = "void"
    name = f"ode_wrapper_{spacetime_name}"
    params = (
        "const double y[9], const commondata_struct *restrict commondata, double f[9]"
    )

    # Construct the body
    body = f"""
    // 1. Declare geometric structs to hold intermediate results
    metric_struct metric;
    connection_struct conn;

    // 2. Compute Metric 
    g4DD_metric_{spacetime_name}(commondata, y, &metric);
    
    // 3. Compute Connections (Christoffel Symbols)
    connections_{spacetime_name}(commondata, y, &conn);
    
    // 4. Compute Geodesic RHS
    calculate_ode_rhs(y, &metric, &conn, f);
    """

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

    import nrpy.infrastructures.BHaH.BHaH_defines_h as Bdefines_h

    sys.path.append(os.getcwd())
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestODEWrapper")

    SPACETIME = "KerrSchild_Cartesian"
    logger.info("Test: Generating Internal Wrapper C-code for %s...", SPACETIME)

    try:
        ode_wrapper(SPACETIME)
        func_name = f"ode_wrapper_{SPACETIME}"
        if func_name in cfc.CFunction_dict:
            filename = f"{func_name}.c"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(cfc.CFunction_dict[func_name].full_function)
            logger.info(" -> Success! Wrote %s", filename)
            Bdefines_h.output_BHaH_defines_h(project_dir=".")
        else:
            raise RuntimeError(f"Function {func_name} not registered.")
    except (RuntimeError, OSError) as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)

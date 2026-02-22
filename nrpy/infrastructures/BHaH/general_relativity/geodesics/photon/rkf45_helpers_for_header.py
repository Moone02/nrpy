"""
Generates the C helper functions for the RKF45 computational kernels.
Updated to support batched processing using a flattened SoA architecture 
and macro-based 1D indexing for optimal SIMT performance.

Author: Dalton J. Moone
"""

from nrpy.infrastructures.BHaH import BHaH_defines_h as Bdefines_h


def rkf45_helpers_for_header(spacetime_name: str) -> None:
    """
    Generate and register the RKF45 kernels for injection into BHaH_defines.h.

    This function generates:
    1. `calculate_rkf45_stage_f_temp()`: Computes the temporary state vector (f_temp)
       for a specific RKF45 stage using local batch indexing.
    2. `rkf45_kernel()`: Performs the final summations to compute the 5th-order
       solution and the error estimate using local batch indexing.

    :param spacetime_name: String used to link to the correct ode_wrapper.
    """
    # The C code block to be injected. Note the escaped {{ and }} for the f-string.
    c_code_for_header = f"""
// =============================================
// NRPy-Generated RKF45 Stepper Helpers
// =============================================

// --- RKF45 Stage State Calculator ---
// Computes the intermediate state vector (f_temp) for a specific RKF45 stage.
static inline void calculate_rkf45_stage_f_temp(
    const int stage, 
    const double *restrict f_in, 
    const double *restrict k_array, 
    const double h, 
    double *restrict f_temp_out, 
    const int batch_size, 
    const int batch_id
) {{
    // Compute the intermediate state (f_temp) using the Butcher Tableau.
    // 2D arrays are flattened to 1D using: IDX_LOCAL(stage * 9 + i, batch_id, batch_size)
    switch (stage) {{
        case 1: // For k1, we evaluate at the initial position
            for (int i = 0; i < 9; ++i) 
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)];
            break;
        case 2: // For k2
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + 
                    h * (1.0/4.0) * k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)];
            }}
            break;
        case 3: // For k3
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + 
                    h * ( (3.0/32.0)*k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] + 
                          (9.0/32.0)*k_array[IDX_LOCAL(1*9 + i, batch_id, batch_size)] );
            }}
            break;
        case 4: // For k4
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + 
                    h * ( (1932.0/2197.0)*k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] - 
                          (7200.0/2197.0)*k_array[IDX_LOCAL(1*9 + i, batch_id, batch_size)] + 
                          (7296.0/2197.0)*k_array[IDX_LOCAL(2*9 + i, batch_id, batch_size)] );
            }}
            break;
        case 5: // For k5
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + 
                    h * ( (439.0/216.0)*k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] - 
                          8.0*k_array[IDX_LOCAL(1*9 + i, batch_id, batch_size)] + 
                          (3680.0/513.0)*k_array[IDX_LOCAL(2*9 + i, batch_id, batch_size)] - 
                          (845.0/4104.0)*k_array[IDX_LOCAL(3*9 + i, batch_id, batch_size)] );
            }}
            break;
        case 6: // For k6
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + 
                    h * ( -(8.0/27.0)*k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] + 
                          2.0*k_array[IDX_LOCAL(1*9 + i, batch_id, batch_size)] - 
                          (3544.0/2565.0)*k_array[IDX_LOCAL(2*9 + i, batch_id, batch_size)] + 
                          (1859.0/4104.0)*k_array[IDX_LOCAL(3*9 + i, batch_id, batch_size)] - 
                          (11.0/40.0)*k_array[IDX_LOCAL(4*9 + i, batch_id, batch_size)] );
            }}
            break;
    }}
}}

// --- RKF45 Kernel ---
// Pure computational kernel for the final RKF45 step summation.
static inline void rkf45_kernel(
    const double *restrict f_in, 
    const double *restrict k_array, 
    const double h, 
    double *restrict f_out, 
    double *restrict f_err, 
    const int batch_size, 
    const int batch_id
) {{
    // Calculate the 4th and 5th order solutions simultaneously.
    for (int i = 0; i < 9; ++i) {{
        // 4th-Order Accurate Solution (for error estimate)
        double f_4th = f_in[IDX_LOCAL(i, batch_id, batch_size)] + h * ( 
                                       (25.0/216.0) * k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] +
                                       (1408.0/2565.0) * k_array[IDX_LOCAL(2*9 + i, batch_id, batch_size)] +
                                       (2197.0/4104.0) * k_array[IDX_LOCAL(3*9 + i, batch_id, batch_size)] -
                                       (1.0/5.0) * k_array[IDX_LOCAL(4*9 + i, batch_id, batch_size)] );

        // 5th-Order Accurate Solution (for final state)
        f_out[IDX_LOCAL(i, batch_id, batch_size)] = f_in[IDX_LOCAL(i, batch_id, batch_size)] + h * ( 
                                   (16.0/135.0) * k_array[IDX_LOCAL(0*9 + i, batch_id, batch_size)] +
                                   (6656.0/12825.0) * k_array[IDX_LOCAL(2*9 + i, batch_id, batch_size)] +
                                   (28561.0/56430.0) * k_array[IDX_LOCAL(3*9 + i, batch_id, batch_size)] -
                                   (9.0/50.0) * k_array[IDX_LOCAL(4*9 + i, batch_id, batch_size)] +
                                   (2.0/55.0) * k_array[IDX_LOCAL(5*9 + i, batch_id, batch_size)] );

        // The local error estimate is the difference between orders.
        f_err[IDX_LOCAL(i, batch_id, batch_size)] = f_out[IDX_LOCAL(i, batch_id, batch_size)] - f_4th;
    }}
}}
"""

    # Register this C code block to be injected into the BHaH_defines.h header file.
    Bdefines_h.register_BHaH_defines("rkf45_helpers", c_code_for_header)

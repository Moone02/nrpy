"""
Generates the C helper functions for the RKF45 computational kernels.
Updated to support batched processing by decoupling state calculation from RHS evaluation.

Author: Dalton J. Moone
"""

from nrpy.infrastructures.BHaH import BHaH_defines_h as Bdefines_h


def rkf45_helpers_for_header(spacetime_name: str) -> None:
    """
    Generate and register the RKF45 kernels for injection into BHaH_defines.h.

    This function generates:
    1. `calculate_rkf45_stage_f_temp()`: Computes the temporary state vector (f_temp)
       for a specific RKF45 stage. It DOES NOT calculate the RHS.
    2. `rkf45_kernel()`: Performs the final summations to compute the 5th-order
       solution and the error estimate.

    :param spacetime_name: String used to link to the correct ode_wrapper.
    """
    # The C code block to be injected.
    c_code_for_header = f"""
// =============================================
// NRPy-Generated RKF45 Stepper Helpers
// =============================================

// --- RKF45 Stage State Calculator ---
// Computes the intermediate state vector (f_temp) for a specific RKF45 stage.
static inline void calculate_rkf45_stage_f_temp(
    const int stage,                // The stage to compute (1-6)
    const double f_in[9],           // The state at the beginning of the step
    const double k_array[6][9],     // Array of the k vectors (previous stages)
    const double h,                 // The step size
    double f_temp_out[9]            // Output: The predicted STATE at this stage
) {{
    // Compute the intermediate state (f_temp) using the Butcher Tableau.
    switch (stage) {{
        case 1: // For k1, we evaluate at the initial position
            for (int i = 0; i < 9; ++i) f_temp_out[i] = f_in[i];
            break;
        case 2: // For k2
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[i] = f_in[i] + h * (1.0/4.0) * k_array[0][i];
            }}
            break;
        case 3: // For k3
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[i] = f_in[i] + h * ( (3.0/32.0)*k_array[0][i] + (9.0/32.0)*k_array[1][i] );
            }}
            break;
        case 4: // For k4
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[i] = f_in[i] + h * ( (1932.0/2197.0)*k_array[0][i] - (7200.0/2197.0)*k_array[1][i] + (7296.0/2197.0)*k_array[2][i] );
            }}
            break;
        case 5: // For k5
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[i] = f_in[i] + h * ( (439.0/216.0)*k_array[0][i] - 8.0*k_array[1][i] + (3680.0/513.0)*k_array[2][i] - (845.0/4104.0)*k_array[3][i] );
            }}
            break;
        case 6: // For k6
            for (int i = 0; i < 9; ++i) {{
                f_temp_out[i] = f_in[i] + h * ( -(8.0/27.0)*k_array[0][i] + 2.0*k_array[1][i] - (3544.0/2565.0)*k_array[2][i] + (1859.0/4104.0)*k_array[3][i] - (11.0/40.0)*k_array[4][i] );
            }}
            break;
    }}
}}

// --- RKF45 Kernel ---
// Pure computational kernel for the final RKF45 step summation.
static inline void rkf45_kernel(
    const double f_in[9],           // The state at the beginning of the step
    const double k_array[6][9],     // Array of the 6 pre-computed k vectors
    const double h,                 // The step size attempted
    double f_out[9],                // Output: the final 5th-order state
    double f_err[9]                 // Output: the error vector (f_5th - f_4th)
) {{
    // Calculate the 4th and 5th order solutions simultaneously.
    for (int i = 0; i < 9; ++i) {{
        // 4th-Order Accurate Solution (for error estimate)
        double f_4th = f_in[i] + h * ( (25.0/216.0) * k_array[0][i] +
                                       (1408.0/2565.0) * k_array[2][i] +
                                       (2197.0/4104.0) * k_array[3][i] -
                                       (1.0/5.0) * k_array[4][i] );

        // 5th-Order Accurate Solution (for final state)
        f_out[i] = f_in[i] + h * ( (16.0/135.0) * k_array[0][i] +
                                   (6656.0/12825.0) * k_array[2][i] +
                                   (28561.0/56430.0) * k_array[3][i] -
                                   (9.0/50.0) * k_array[4][i] +
                                   (2.0/55.0) * k_array[5][i] );

        // The local error estimate is the difference between orders.
        f_err[i] = f_out[i] - f_4th;
    }}
}}
"""

    # Register this C code block to be injected into the BHaH_defines.h header file.
    Bdefines_h.register_BHaH_defines("rkf45_helpers", c_code_for_header)

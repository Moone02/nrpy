"""
Register C function for event finding and state interpolation.

This module registers the 'find_event_time_and_state' C function, which
uses a numerically stable quadratic root-finder to determine the precise
time of an event (e.g., a plane crossing) and then uses second-order
Lagrange polynomials to interpolate the full state vector to that time.

Note: The required 'event_data_struct' and 'event_function_t' types
must be defined in BHaH_defines.h by an external script (e.g., the
top-level integrator) before this function is compiled.

Author: Dalton J. Moone
"""

# Step 0.a: Import standard Python modules
import sys

# Step 0.b: Import NRPy core modules
import nrpy.c_function as cfc


def find_event_time_and_state() -> None:
    """
    Generate and register the generic, robust Lagrange interpolation engine.

    Registers the 'find_event_time_and_state' C function.
    """
    # Step 1: Define C function metadata
    # Per project standards, define local variables for all register_CFunction args.
    includes = ["BHaH_defines.h", "<math.h>"]
    desc = r"""@brief Finds the root of a generic event using a robust, second-order interpolation.

    This engine first finds the precise affine parameter `λ_event` of a crossing
    by fitting a quadratic polynomial to the event function values at three
    consecutive time steps and finding its root. It then uses Lagrange
    polynomials to interpolate the full 9-component state vector to `λ_event`.
    Includes fallbacks to linear interpolation for numerical stability.

    @param[in]  f_prev, f_curr, f_next  State vectors at three consecutive steps.
    @param[in]  lambda_prev, lambda_curr, lambda_next Affine parameters for the three steps.
    @param[in]  event_func          A function pointer to the event function (e.g., distance to a plane).
    @param[in]  event_params        A void pointer to parameters for the event function.
    @param[out] result              Pointer to the event_data_struct to be filled with results.
    """
    name = "find_event_time_and_state"
    params = """const double f_prev[9], const double f_curr[9], const double f_next[9],
                double lambda_prev, double lambda_curr, double lambda_next,
                event_function_t event_func, void *event_params,
                event_data_struct *restrict result"""

    # Step 2: Define the C Body
    # The body is algorithmic, not symbolic, so it is defined as a raw C string.
    body = r"""
    double t0 = lambda_prev, t1 = lambda_curr, t2 = lambda_next;
    double f0 = event_func(f_prev, event_params);
    double f1 = event_func(f_curr, event_params);
    double f2 = event_func(f_next, event_params);

    // --- Step 1: Use linear interpolation as a robust fallback ---
    double t_linear;
    if (f0 * f1 < 0.0 && fabs(f1 - f0) > 1e-12) { // Sign change is between prev and curr
        t_linear = (f1 * t0 - f0 * t1) / (f1 - f0);
    } else if (f1 * f2 < 0.0 && fabs(f2 - f1) > 1e-12) { // Sign change is between curr and next
        t_linear = (f2 * t1 - f1 * t2) / (f2 - f1);
    } else {
        // This can happen if f1 is exactly zero or no sign change is bracketed.
        t_linear = t1;
    }

    // --- Step 2: Attempt quadratic interpolation (Muller's method variant) for higher accuracy ---
    double h0 = t1 - t0;
    double h1 = t2 - t1;

    // Check for degenerate intervals to prevent division by zero.
    if (fabs(h0) < 1e-15 || fabs(h1) < 1e-15 || fabs(h0 + h1) < 1e-15) {
        result->lambda_event = t_linear;
    } else {
        double delta0 = (f1 - f0) / h0;
        double delta1 = (f2 - f1) / h1;
        double a = (delta1 - delta0) / (h1 + h0);
        double b = a * h1 + delta1;
        double c = f2;
        double discriminant = b*b - 4*a*c;

        if (discriminant < 0.0 || fabs(a) < 1e-15) {
            // Discriminant is negative or equation is effectively linear; fallback to linear.
            result->lambda_event = t_linear;
        } else {
            // Use the more numerically stable form of the quadratic formula.
            double denom = (b >= 0.0) ? (b + sqrt(discriminant)) : (b - sqrt(discriminant));
            if (fabs(denom) < 1e-15) {
                result->lambda_event = t_linear;
            } else {
                double t_quad = t2 - (2.0 * c / denom);
                // Only accept the quadratic result if it's within the bracketing interval.
                if (t_quad > fmin(t0, t2) && t_quad < fmax(t0, t2)) {
                    result->lambda_event = t_quad;
                } else {
                    result->lambda_event = t_linear;
                }
            }
        }
    }

    // --- Step 3: Perform final interpolation on the state vector using the found time ---
    double t = result->lambda_event;

    // Check for degenerate intervals again before final interpolation.
    if (fabs(t0 - t1) < 1e-15 || fabs(t0 - t2) < 1e-15 || fabs(t1 - t2) < 1e-15) {
        // Fallback to linear interpolation for the state vector as well.
        double frac = 0.5;
        if (fabs(t2 - t1) > 1e-15) {
            frac = (t - t1) / (t2 - t1);
        }
        for (int i = 0; i < 9; i++) {
            result->f_event[i] = f_curr[i] + frac * (f_next[i] - f_curr[i]);
        }
    } else {
        // Perform full second-order Lagrange polynomial interpolation.
        double L0 = ((t - t1) * (t - t2)) / ((t0 - t1) * (t0 - t2));
        double L1 = ((t - t0) * (t - t2)) / ((t1 - t0) * (t1 - t2));
        double L2 = ((t - t0) * (t - t1)) / ((t2 - t0) * (t2 - t1));
        for (int i = 0; i < 9; i++) {
            result->f_event[i] = f_prev[i] * L0 + f_curr[i] * L1 + f_next[i] * L2;
        }
    }

    result->t_event = result->f_event[0];
    result->found = true;
    """

    # Step 3: Register the C function.
    cfc.register_CFunction(
        includes=includes, desc=desc, name=name, params=params, body=body
    )


if __name__ == "__main__":
    import logging
    import os

    # Ensure local modules can be imported
    sys.path.append(os.getcwd())

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestFindEventTimeAndState")

    logger.info("Test: Generating C-code for find_event_time_and_state...")

    try:
        # 1. Run the Generator
        logger.info(" -> Calling find_event_time_and_state()...")
        find_event_time_and_state()

        # 2. Validation
        cfunc_name = "find_event_time_and_state"

        # Check C Function Registration
        if cfunc_name not in cfc.CFunction_dict:
            raise RuntimeError(
                f"FAIL: '{cfunc_name}' was not registered in cfc.CFunction_dict."
            )

        logger.info(" -> PASS: '%s' function registered successfully.", cfunc_name)

        # 3. Output Files to Current Directory
        for func_name, c_function in cfc.CFunction_dict.items():
            filename = f"{func_name}.c"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(c_function.full_function)
            logger.info("    ... Wrote %s", filename)

        logger.info(" -> Success! C function generated.")

    except Exception as e:
        logger.error(" -> FAIL: find_event_time_and_state test failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

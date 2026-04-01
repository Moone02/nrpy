"""
Generates the main() C function for the Project Singularity-Axiom pipeline.

This module acts as the master orchestrator, handling the initialization of
global simulation parameters, memory allocation for trajectory results, and
the final serialization of the light blueprint to disk using a tiled approach.

Author: Dalton J. Moone
"""

import logging
import sys
import nrpy.c_function as cfc
import nrpy.params as par

# Configure logging to adhere to static analysis and project standards
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(spacetime_name: str) -> None:
    """
    Register the master orchestrator C function for the geodesic integrator.
    
    This version implements Spatial Domain Decomposition (Tiling) to allow 
    high-resolution renders with low memory overhead.
    """
    
    # 1. Register Tiling Parameters
    par.register_CodeParameter("int", __name__, "window_tiles_width", 1, commondata=True, add_to_parfile=True)
    par.register_CodeParameter("int", __name__, "window_tiles_height", 1, commondata=True, add_to_parfile=True)

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "stdio.h", "stdlib.h", "math.h"]

    desc = f"""@brief Master entry point for the Project Singularity-Axiom integrator.

    Algorithm:
    1. Initializes core data structures and sets default physical constants.
    2. Parses command-line arguments and parameter files to override defaults.
    3. Calculates the orthonormal basis (nx, ny, nz) for the camera window.
    4. Loops through a grid of tiles (tx, ty), shifting the window center.
    5. Dispatches the batched numerical integrator for the {spacetime_name} metric per tile.
    6. Serializes and zips each tile's data to 'light_blueprint_XX_YY.zip'."""

    cfunc_type = "int"
    name = "main"
    params = "int argc, const char *argv[]"

    # 2. Build the C body
    body = f"""
    // --- CORE STRUCTURE INITIALIZATION ---
    // Populates the master structure with the base metric parameters.
    commondata_struct commondata; 

    // --- PARAMETER PARSING ---
    // Sets default metric properties and overrides them based on host system input.
    commondata_struct_set_to_default(&commondata); 
    cmdline_input_and_parfile_parser(&commondata, argc, argv); 

    // --- INITIALIZE DYNAMIC WINDOW CENTER ---
    // The tile orchestrator modifies the window center dynamically per tile. 
    // We seed the active window center with the original master center provided by the configuration.
    commondata.window_center_x = commondata.original_window_center_x;
    commondata.window_center_y = commondata.original_window_center_y;
    commondata.window_center_z = commondata.original_window_center_z;

    // --- TELEMETRY AND PARAMETER VERIFICATION ---
    // Outputs core physical and architectural variables to the standard output.
    printf("=============================================\\n");
    printf("  Project Singularity-Axiom: Geodesic Engine  \\n");
    printf("=============================================\\n");
    printf("Spacetime Metric: {spacetime_name}\\n");
    
    printf("--- Analytic Spacetimes ---\\n");
    printf("Mass Scale (M): %.3f\\n", commondata.M_scale);
    printf("Spin Parameter (a): %.3f\\n", commondata.a_spin);

    printf("--- Initial Conditions ---\\n");
    printf("Scan Density: %d\\n", commondata.scan_density);
    printf("Start Time (t_start): %.3f\\n", commondata.t_start);

    printf("--- Batch Integrator Numerical ---\\n");
    printf("Initial Step Size (h_initial): %.3f\\n", commondata.numerical_initial_h);
    printf("Max Momentum (p_t,max): %.3f\\n", commondata.p_t_max);
    printf("Conservation Check: %d\\n", commondata.perform_conservation_check);
    printf("Escape Radius (r_escape): %.3f\\n", commondata.r_escape);
    printf("Slot Delta t: %.3f\\n", commondata.slot_manager_delta_t);
    printf("Slot Min t: %.3f\\n", commondata.slot_manager_t_min);
    printf("Integration Max t: %.3f\\n", commondata.t_integration_max);

    printf("--- RKF45 Control ---\\n");
    printf("Absolute Tolerance: %e\\n", commondata.rkf45_absolute_error_tolerance);
    printf("Relative Tolerance: %e\\n", commondata.rkf45_error_tolerance);
    printf("Max Step (h_max): %.3f\\n", commondata.rkf45_h_max);
    printf("Min Step (h_min): %e\\n", commondata.rkf45_h_min);
    printf("Max Retries: %d\\n", commondata.rkf45_max_retries);
    printf("Safety Factor: %.3f\\n", commondata.rkf45_safety_factor);

    printf("--- Source Plane ---\\n");
    printf("Center (x, y, z): %.3f, %.3f, %.3f\\n", commondata.source_plane_center_x, commondata.source_plane_center_y, commondata.source_plane_center_z);
    printf("Normal (x, y, z): %.3f, %.3f, %.3f\\n", commondata.source_plane_normal_x, commondata.source_plane_normal_y, commondata.source_plane_normal_z);
    printf("Max Radius (r_max): %.3f\\n", commondata.source_r_max);
    printf("Min Radius (r_min): %.3f\\n", commondata.source_r_min);
    printf("Up Vector (x, y, z): %.3f, %.3f, %.3f\\n", commondata.source_up_vec_x, commondata.source_up_vec_y, commondata.source_up_vec_z);

    printf("--- Window Plane ---\\n");
    printf("Camera Pos (x, y, z): %.3f, %.3f, %.3f\\n", commondata.camera_pos_x, commondata.camera_pos_y, commondata.camera_pos_z);
    printf("Original Window Center (x, y, z): %.3f, %.3f, %.3f\\n", commondata.original_window_center_x, commondata.original_window_center_y, commondata.original_window_center_z);
    printf("Window Height: %.3f\\n", commondata.window_height);
    printf("Window Width: %.3f\\n", commondata.window_width);
    printf("Window Up Vec (x, y, z): %.3f, %.3f, %.3f\\n", commondata.window_up_vec_x, commondata.window_up_vec_y, commondata.window_up_vec_z);
    printf("Resolution Grid: %d x %d tiles, each %d x %d px\\n", 
           commondata.window_tiles_width, commondata.window_tiles_height, 
           commondata.scan_density, commondata.scan_density);

    // --- TILING CALCULATIONS ---
    // Master parameters define the full global frame, referencing the immutable original center.
    const double master_center_x = commondata.original_window_center_x;
    const double master_center_y = commondata.original_window_center_y;
    const double master_center_z = commondata.original_window_center_z;
    const double master_width    = commondata.window_width;
    const double master_height   = commondata.window_height;

    const double tile_w = master_width  / (double)commondata.window_tiles_width;
    const double tile_h = master_height / (double)commondata.window_tiles_height;

    // --- BASIS VECTOR CALCULATION ---
    // n_z: The forward "Look" vector
    double n_z[3] = {{master_center_x - commondata.camera_pos_x, 
                     master_center_y - commondata.camera_pos_y, 
                     master_center_z - commondata.camera_pos_z}};
    double mag_n_z = sqrt(n_z[0]*n_z[0] + n_z[1]*n_z[1] + n_z[2]*n_z[2]);
    for(int j=0; j<3; j++) n_z[j] /= mag_n_z;

    // n_x: The "Right" vector (Horizontal)
    // Calculated as Up (guide_up) x Forward (n_z) to point Right.
    const double guide_up[3] = {{commondata.window_up_vec_x, commondata.window_up_vec_y, commondata.window_up_vec_z}};
    double n_x[3] = {{guide_up[1]*n_z[2] - guide_up[2]*n_z[1], 
                     guide_up[2]*n_z[0] - guide_up[0]*n_z[2], 
                     guide_up[0]*n_z[1] - guide_up[1]*n_z[0]}};
    double mag_n_x = sqrt(n_x[0]*n_x[0] + n_x[1]*n_x[1] + n_x[2]*n_x[2]);
    
    // Fallback logic for near-nadir camera angles (looking straight up or down)
    if (mag_n_x < 1e-10) {{
        double alt_up[3] = {{0.0, 1.0, 0.0}};
        if (fabs(n_z[1]) > 0.9) {{ alt_up[1] = 0.0; alt_up[2] = 1.0; }}
        n_x[0] = alt_up[1]*n_z[2] - alt_up[2]*n_z[1];
        n_x[1] = alt_up[2]*n_z[0] - alt_up[0]*n_z[2];
        n_x[2] = alt_up[0]*n_z[1] - alt_up[1]*n_z[0];
        mag_n_x = sqrt(n_x[0]*n_x[0] + n_x[1]*n_x[1] + n_x[2]*n_x[2]);
    }}
    for(int j=0; j<3; j++) n_x[j] /= mag_n_x;

    // n_y: The "Up" vector (Vertical)
    // Calculated as Forward (n_z) x Right (n_x) to point Up.
    double n_y[3] = {{n_z[1]*n_x[2] - n_z[2]*n_x[1], 
                     n_z[2]*n_x[0] - n_z[0]*n_x[2], 
                     n_z[0]*n_x[1] - n_z[1]*n_x[0]}};
    double mag_n_y = sqrt(n_y[0]*n_y[0] + n_y[1]*n_y[1] + n_y[2]*n_y[2]);
    for(int j=0; j<3; j++) n_y[j] /= mag_n_y;

    // --- RESOURCE ALLOCATION ---
    const long int num_rays = (long int)commondata.scan_density * commondata.scan_density; 
    blueprint_data_t *restrict results_buffer = (blueprint_data_t *)malloc(sizeof(blueprint_data_t) * num_rays);

    if (results_buffer == NULL) {{
        fprintf(stderr, "FATAL: Failed to allocate %ld rays. Check system RAM.\\n", num_rays);
        exit(1);
    }}
    // --- NESTED TILING LOOP ---
    for (int ty = 0; ty < commondata.window_tiles_height; ty++) {{
        for (int tx = 0; tx < commondata.window_tiles_width; tx++) {{
            
            // 1. Update Tile Dimensions
            commondata.window_width  = tile_w;
            commondata.window_height = tile_h;

            // 2. Calculate Local Tile Center
            double offset_x = -master_width/2.0  + (tx + 0.5) * tile_w;
            double offset_y = -master_height/2.0 + (ty + 0.5) * tile_h;

            commondata.window_center_x = master_center_x + offset_x * n_x[0] + offset_y * n_y[0];
            commondata.window_center_y = master_center_y + offset_x * n_x[1] + offset_y * n_y[1];
            commondata.window_center_z = master_center_z + offset_x * n_x[2] + offset_y * n_y[2];

            printf("[Tile %02d,%02d] Processing at Center (%.3f, %.3f, %.3f)...\\n", 
                    tx, ty, commondata.window_center_x, commondata.window_center_y, commondata.window_center_z);

            // 3. Execute Numerical Integration Pipeline
            batch_integrator_numerical(&commondata, num_rays, results_buffer);

            // 3.5. Coordinate Global Shift
            // Maps local tile-space hits to the global window coordinate system.
            // y_w (horizontal) corresponds to offset_x along n_x.
            // z_w (vertical) corresponds to offset_y along n_y.
            for (long int i = 0; i < num_rays; i++) {{
                results_buffer[i].y_w += offset_x;
                results_buffer[i].z_w += offset_y;
            }}

            // 4. Data Serialization
            char bin_name[256], zip_cmd[512];
            sprintf(bin_name, "light_blueprint_%02d_%02d.bin", tx, ty);

            FILE *fp = fopen(bin_name, "wb");
            if (fp) {{
                fwrite(results_buffer, sizeof(blueprint_data_t), num_rays, fp);
                fclose(fp);

                // 5. Linux-Native Compression with move flag (-m) and Success Check
                sprintf(zip_cmd, "zip -mq light_blueprint_%02d_%02d.zip %s", tx, ty, bin_name);
                int status = system(zip_cmd);
                if (status != 0) {{
                    fprintf(stderr, "ERROR: Compression failed for %s. Keeping raw binary.\\n", bin_name);
                }}
            }} else {{
                fprintf(stderr, "ERROR: Could not write to %s\\n", bin_name);
                free(results_buffer);
                exit(1);
            }}
        }}
    }}

    // --- FINAL CLEANUP & SHUTDOWN ---
    free(results_buffer);
    printf("Simulation successful. Data stored in zipped tiles.\\n");
    return 0;
    """

    # 3. Register the function with the NRPy environment
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
    )


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()
    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
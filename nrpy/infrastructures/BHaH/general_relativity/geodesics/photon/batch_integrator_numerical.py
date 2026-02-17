"""
Generates the main C orchestrator for the numerical integration pipeline.

Author: Dalton J. Moone
"""

import nrpy.c_function as cfc


def batch_integrator_numerical(spacetime_name: str) -> None:
    """
    Generate and register the C orchestrator for the numerical pipeline.

    This function generates the C code for `batch_integrator_numerical()`, which
    is the top-level orchestrator for the entire simulation when using the
    numerical pipeline. It implements the "Iterative Time Slotting" algorithm,
    manages the main epoch loop, and orchestrates the event cascade for each
    photon at every step.

    :param spacetime_name: The name of the spacetime (e.g. "KerrSchild_Cartesian")
                           used to link to specific helper functions.
    """
    # Per project standards, define local variables for all register_CFunction args.
    # Standard headers are included automatically by BHaH_defines.h.
    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
        "omp.h",
    ]

    desc = r"""@brief Main orchestrator for the numerical pipeline with debugging and conservation features.

    This function implements the "Iterative Time Slotting" algorithm. It manages
    the main epoch loop, processes photons in parallel bundles, calls the adaptive
    RKF45 stepper for each photon, and orchestrates the full event cascade
    (termination checks, plane crossings) after each step.

    @param[in] commondata         Pointer to commondata struct with runtime parameters.
    @param[in] num_rays           The total number of rays to integrate.
    @param[out] results_buffer    A pointer to the buffer to be filled with final blueprint data.
    """

    name = "batch_integrator_numerical"

    params = """
    const commondata_struct *restrict commondata,
    long int num_rays,
    blueprint_data_t *results_buffer
    """

    # We construct the function names dynamically based on the spacetime input.
    # Note: 'photon' is hardcoded here as this is the photon integrator.
    initial_conditions_func = f"set_initial_conditions_cartesian_{spacetime_name}"
    interpolation_func = f"placeholder_interpolation_engine_{spacetime_name}"
    conserved_quantities_func = f"conserved_quantities_{spacetime_name}_photon"

    # The body is algorithmic, not symbolic, so it is defined as a raw C string.
    body = f"""
    // === INITIALIZATION ===
    printf("Initializing %ld photon states for NUMERICAL integration...\\n", num_rays);
    PhotonState *all_photons = (PhotonState *)malloc(sizeof(PhotonState) * num_rays);
    if (all_photons == NULL) {{ fprintf(stderr, "Error: Failed to allocate memory for photon states.\\n"); exit(1); }}
    long int active_photons = num_rays;

    // --- Debug Mode File Setup ---
    FILE *fp_debug = NULL;
    if (commondata->debug_mode) {{
        fp_debug = fopen("photon_path_numerical.txt", "w");
        if (fp_debug == NULL) {{ exit(1); }}
        fprintf(fp_debug, "# affine_param\\tt\\tx\\ty\\tz\\tp_t\\tp_x\\tp_y\\tp_z\\tL\\n");
    }}

    // --- Conservation Check Memory Allocation ---
    // Stores [E, Lx, Ly, Lz, Q] for each ray
    double (*initial_conserved_quantities)[5] = NULL;
    if (commondata->perform_conservation_check) {{
        initial_conserved_quantities = malloc(num_rays * 5 * sizeof(double));
        if(initial_conserved_quantities == NULL) {{ exit(1); }}
    }}

    // --- Time Slot Manager and Camera Geometry Setup ---
    TimeSlotManager tsm;
    slot_manager_init(&tsm, commondata->slot_manager_t_min, commondata->t_start + 1.0, commondata->slot_manager_delta_t);
    const double camera_pos[3] = {{commondata->camera_pos_x, commondata->camera_pos_y, commondata->camera_pos_z}};
    const double window_center[3] = {{commondata->window_center_x, commondata->window_center_y, commondata->window_center_z}};
    double n_z[3] = {{window_center[0] - camera_pos[0], window_center[1] - camera_pos[1], window_center[2] - camera_pos[2]}};
    double mag_n_z = sqrt(SQR(n_z[0]) + SQR(n_z[1]) + SQR(n_z[2]));
    for(int i=0; i<3; i++) n_z[i] /= mag_n_z;
    const double guide_up[3] = {{commondata->window_up_vec_x, commondata->window_up_vec_y, commondata->window_up_vec_z}};
    double n_x[3] = {{n_z[1]*guide_up[2] - n_z[2]*guide_up[1], n_z[2]*guide_up[0] - n_z[0]*guide_up[2], n_z[0]*guide_up[1] - n_z[1]*guide_up[0]}};
    double mag_n_x = sqrt(SQR(n_x[0]) + SQR(n_x[1]) + SQR(n_x[2]));
    if (mag_n_x < 1e-9) {{
        double alternative_up[3] = {{0.0, 1.0, 0.0}};
        if (fabs(n_z[1]) > 0.999) {{ alternative_up[1] = 0.0; alternative_up[2] = 1.0; }}
        n_x[0] = alternative_up[1]*n_z[2] - alternative_up[2]*n_z[1];
        n_x[1] = alternative_up[2]*n_z[0] - alternative_up[0]*n_z[2];
        n_x[2] = alternative_up[0]*n_z[1] - alternative_up[1]*n_z[0];
        mag_n_x = sqrt(SQR(n_x[0]) + SQR(n_x[1]) + SQR(n_x[2]));
    }}
    for(int i=0; i<3; i++) n_x[i] /= mag_n_x;
    double n_y[3] = {{n_z[1]*n_x[2] - n_z[2]*n_x[1], n_z[2]*n_x[0] - n_z[0]*n_x[2], n_z[0]*n_x[1] - n_z[1]*n_x[0]}};

    // --- Set Initial Conditions for All Photons in Parallel ---
    #pragma omp parallel for
    for (long int i = 0; i < num_rays; i++) {{
        const int j = i / commondata->scan_density;
        const int k = i % commondata->scan_density;
        const double x_pix = -commondata->window_size/2.0 + (k + 0.5) * (commondata->window_size / commondata->scan_density);
        const double y_pix = -commondata->window_size/2.0 + (j + 0.5) * (commondata->window_size / commondata->scan_density);
        double target_pos[3] = {{window_center[0] + x_pix*n_x[0] + y_pix*n_y[0],
                                 window_center[1] + x_pix*n_x[1] + y_pix*n_y[1],
                                 window_center[2] + x_pix*n_x[2] + y_pix*n_y[2]}};

        {initial_conditions_func}(commondata, target_pos, all_photons[i].f);

        all_photons[i].affine_param = 0.0;
        all_photons[i].h = commondata->numerical_initial_h;
        all_photons[i].status = ACTIVE;
        all_photons[i].rejection_retries = 0;

        for(int ii=0; ii<9; ++ii) {{
            all_photons[i].f_p[ii] = all_photons[i].f[ii];
            all_photons[i].f_p_p[ii] = all_photons[i].f[ii];
        }}
        all_photons[i].affine_param_p = all_photons[i].affine_param;
        all_photons[i].affine_param_p_p = all_photons[i].affine_param;

        // Initial Plane Checks
        plane_event_params window_params = {{
            {{n_z[0], n_z[1], n_z[2]}},
            n_z[0]*window_center[0] + n_z[1]*window_center[1] + n_z[2]*window_center[2]
        }};
        all_photons[i].on_positive_side_of_window_prev = (plane_event_func(all_photons[i].f, &window_params) > 0);

        plane_event_params source_params = {{
            {{commondata->source_plane_normal_x, commondata->source_plane_normal_y, commondata->source_plane_normal_z}},
            commondata->source_plane_center_x*commondata->source_plane_normal_x +
            commondata->source_plane_center_y*commondata->source_plane_normal_y +
            commondata->source_plane_center_z*commondata->source_plane_normal_z
        }};
        all_photons[i].on_positive_side_of_source_prev = (plane_event_func(all_photons[i].f, &source_params) > 0);

        all_photons[i].source_event_data.found = false;
        all_photons[i].window_event_data.found = false;
    }}

    // --- Perform Initial Conservation Checks ---
    if (commondata->perform_conservation_check) {{
        printf("Performing initial conservation checks for all %ld rays...\\n", num_rays);
        #pragma omp parallel for
        for (long int i = 0; i < num_rays; ++i) {{
            // UPDATED: Use dynamic function name and pointer signature for conserved quantities
            {conserved_quantities_func}(commondata, all_photons[i].f,
                                        &initial_conserved_quantities[i][0], &initial_conserved_quantities[i][1],
                                        &initial_conserved_quantities[i][2], &initial_conserved_quantities[i][3],
                                        &initial_conserved_quantities[i][4]);
        }}
    }}

    // --- Place All Photons in the Initial Time Slot ---
    int initial_slot_idx = slot_get_index(&tsm, commondata->t_start);
    if(initial_slot_idx != -1) {{
        for(long int i=0; i<num_rays; ++i) {{ slot_add_photon(&tsm.slots[initial_slot_idx], i); }}
    }} else {{
        fprintf(stderr, "Error: Initial t_start is outside the defined time slot manager domain.\\n");
        exit(1);
    }}

    printf("Starting NUMERICAL batch integration loop (Strictly Photons)...\\n");

    // --- Allocate Buffers for Batch Processing ---
    photon_request_t *requests = (photon_request_t *)malloc(sizeof(photon_request_t) * BUNDLE_CAPACITY);
    metric_struct *metric_results = (metric_struct *)malloc(sizeof(metric_struct) * BUNDLE_CAPACITY);
    connection_struct *christoffel_results = (connection_struct *)malloc(sizeof(connection_struct) * BUNDLE_CAPACITY);
    if (!requests || !metric_results || !christoffel_results) {{ exit(1); }}

    double start_time = omp_get_wtime();
    long int initial_active_photons = active_photons;

    // === MAIN EPOCH LOOP: Iterate backwards through time slots ===
    for (int i = initial_slot_idx; i >= 0 && active_photons > 0; i--) {{
        while (tsm.slots[i].count > 0) {{
            long int bundle_size = MIN(tsm.slots[i].count, BUNDLE_CAPACITY);
            long int bundle_photons[bundle_size];
            slot_remove_chunk(&tsm.slots[i], bundle_photons, bundle_size);

            // --- Inner Rejection-Retry Loop ---
            long int needs_retry_indices[bundle_size];
            long int needs_retry_count = bundle_size;
            for(long int j=0; j<bundle_size; ++j) needs_retry_indices[j] = bundle_photons[j];

            while (needs_retry_count > 0) {{
                // Buffers for RKF45 intermediate steps
                double (*k_array)[6][9] = malloc(needs_retry_count * 6 * 9 * sizeof(double));
                double (*f_start)[9] = malloc(needs_retry_count * 9 * sizeof(double));
                if (!k_array || !f_start) {{ fprintf(stderr, "Error: Failed to allocate retry buffers.\\n"); exit(1); }}

                // Initialize f_start for this retry batch
                for(long int j=0; j<needs_retry_count; ++j) {{
                    long int photon_idx = needs_retry_indices[j];
                    for(int k=0; k<9; ++k) f_start[j][k] = all_photons[photon_idx].f[k];
                }}

                // --- RKF45 Stepper: 6 stages ---
                for (int stage = 1; stage <= 6; ++stage) {{

                    // 1. Calculate f_temp (Proposed Position) for this stage
                    #pragma omp parallel for
                    for (long int j = 0; j < needs_retry_count; ++j) {{
                        long int photon_idx = needs_retry_indices[j];
                        double h = all_photons[photon_idx].h;
                        double f_temp[9] = {{0}}; // Add {0} to initialize the array to zero

                        // Use the helper to calculate position.
                        calculate_rkf45_stage_f_temp(stage, f_start[j], k_array[j], h, f_temp);

                        // Fill request buffer for batch interpolation
                        requests[j].photon_id = photon_idx;
                        for(int k=0; k<4; k++) requests[j].pos[k] = f_temp[k];
                    }}

                    // 2. Batch call to the interpolation engine
                    {interpolation_func}(commondata, needs_retry_count, requests, metric_results, christoffel_results);

                    // 3. Calculate RHS (k-vector) using interpolated geometry
                    #pragma omp parallel for
                    for (long int j = 0; j < needs_retry_count; ++j) {{
                        long int photon_idx = needs_retry_indices[j];
                        double f_temp[9] = {{0}}; // Add {0} to initialize the array to zero
                        double h = all_photons[photon_idx].h;

                        // Re-calculate f_temp locally
                        calculate_rkf45_stage_f_temp(stage, f_start[j], k_array[j], h, f_temp);

                        // Compute derivative (RHS) and store in k_array
                        calculate_ode_rhs(f_temp, &metric_results[j], &christoffel_results[j], k_array[j][stage-1]);
                    }}
                }}

                long int current_retry_count = needs_retry_count;
                needs_retry_count = 0;

                // --- Step Acceptance and Control ---
                #pragma omp parallel for
                for (long int j = 0; j < current_retry_count; ++j) {{
                    long int photon_idx = needs_retry_indices[j];
                    double f_out[9], f_err[9];

                    rkf45_kernel(f_start[j], k_array[j], all_photons[photon_idx].h, f_out, f_err);

                    bool step_accepted = update_photon_state_and_stepsize(&all_photons[photon_idx], f_start[j], f_out, f_err, commondata);

                    if (step_accepted) {{
                        // Update history
                        for(int k=0; k<9; ++k) {{
                            all_photons[photon_idx].f_p_p[k] = all_photons[photon_idx].f_p[k];
                            all_photons[photon_idx].f_p[k] = f_start[j][k];
                        }}
                        all_photons[photon_idx].affine_param_p_p = all_photons[photon_idx].affine_param_p;
                        all_photons[photon_idx].affine_param_p = all_photons[photon_idx].affine_param - all_photons[photon_idx].h;

                        if (commondata->debug_mode && photon_idx == 0) {{
                            #pragma omp critical
                            {{
                                const double *f = all_photons[photon_idx].f;
                                fprintf(fp_debug, "%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\t%.15e\\n",
                                        all_photons[photon_idx].affine_param,
                                        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
                            }}
                        }}
                    }} else {{
                        if (all_photons[photon_idx].rejection_retries > commondata->rkf45_max_retries) {{
                            all_photons[photon_idx].status = FAILURE_RKF45_REJECTION_LIMIT;
                        }} else {{
                            #pragma omp critical
                            {{ needs_retry_indices[needs_retry_count++] = photon_idx; }}
                        }}
                    }}
                }}
                free(k_array); 
                free(f_start);
            }} // End rejection-retry while loop

            // --- Event Cascade and Re-slotting ---
            for (long int j = 0; j < bundle_size; ++j) {{
                long int photon_idx = bundle_photons[j];
                if (all_photons[photon_idx].status == ACTIVE) {{
                    const double p_t = all_photons[photon_idx].f[4];
                    const double r_sq = SQR(all_photons[photon_idx].f[1]) + SQR(all_photons[photon_idx].f[2]) + SQR(all_photons[photon_idx].f[3]);

                    if (fabs(p_t) > commondata->p_t_max) {{
                        all_photons[photon_idx].status = FAILURE_PT_TOO_BIG;
                    }} else if (fabs(all_photons[photon_idx].f[0]) > commondata->t_integration_max) {{
                        all_photons[photon_idx].status = FAILURE_T_MAX_EXCEEDED;
                    }} else if (r_sq > SQR(commondata->r_escape)) {{
                        all_photons[photon_idx].status = TERMINATION_TYPE_CELESTIAL_SPHERE;
                    }} else {{
                        // Check for geometric plane intersections
                        if (all_photons[photon_idx].status == ACTIVE) {{
                            event_detection_manager(all_photons[photon_idx].f_p_p, all_photons[photon_idx].f_p, all_photons[photon_idx].f,
                                                    all_photons[photon_idx].affine_param_p_p, all_photons[photon_idx].affine_param_p, all_photons[photon_idx].affine_param,
                                                    commondata, &all_photons[photon_idx].on_positive_side_of_window_prev, &all_photons[photon_idx].on_positive_side_of_source_prev,
                                                    &all_photons[photon_idx].window_event_data, &all_photons[photon_idx].source_event_data);
                            if (all_photons[photon_idx].source_event_data.found) {{
                                blueprint_data_t temp_blueprint;
                                if (handle_source_plane_intersection(&all_photons[photon_idx].source_event_data, commondata, &temp_blueprint)) {{
                                    all_photons[photon_idx].status = TERMINATION_TYPE_SOURCE_PLANE;
                                }} else {{
                                    all_photons[photon_idx].source_event_data.found = false; // Reset if hit was invalid
                                }}
                            }}
                        }}
                    }}
                }}

                if (all_photons[photon_idx].status != ACTIVE) {{
                    #pragma omp atomic
                    active_photons--;
                }} else {{
                    int new_slot_idx = slot_get_index(&tsm, all_photons[photon_idx].f[0]);
                    if (new_slot_idx != -1) {{
                        slot_add_photon(&tsm.slots[new_slot_idx], photon_idx);
                    }} else {{
                        all_photons[photon_idx].status = FAILURE_SLOT_MANAGER_ERROR;
                        #pragma omp atomic
                        active_photons--;
                    }}
                }}
            }}
        }} // End while(tsm.slots[i].count > 0)

        // Progress reporting (Time-Throttled)
        #pragma omp master
        {{
            static double last_update_time = 0.0;
            double current_time = omp_get_wtime();
            if (current_time - last_update_time > 0.1) {{
                double elapsed_time = current_time - start_time;
                long int photons_terminated = initial_active_photons - active_photons;
                double rays_per_sec = (elapsed_time > 1e-9) ? (double)photons_terminated / elapsed_time : 0.0;
                double percent_done = (double)photons_terminated / initial_active_photons * 100.0;
                printf("\\rEpoch (Slot %d), Active Photons: %ld (%.1f%% done, %.1f rays/sec)      ", i, active_photons, percent_done, rays_per_sec);
                fflush(stdout);
                last_update_time = current_time;
            }}
        }}
    }} // End main epoch loop

    // --- Final Progress Update ---
    #pragma omp master
    {{
        double final_time = omp_get_wtime();
        double total_elapsed = final_time - start_time;
        double final_rays_per_sec = (total_elapsed > 1e-9) ? (double)initial_active_photons / total_elapsed : 0.0;
        printf("\\rEpoch (Slot 0), Active Photons: 0 (100.0%% done, %.1f rays/sec)           \\n", final_rays_per_sec);
        fflush(stdout);
    }}

    printf("Batch integration finished.\\n");

    if (commondata->debug_mode && fp_debug != NULL) {{ fclose(fp_debug); }}

    // --- FINAL CONSERVATION CHECK ---
    if (commondata->perform_conservation_check) {{
        printf("\\n--- Conservation Check Summary ---\\n");
        double max_dE = 0.0, max_dL = 0.0, max_dQ = 0.0;
        long int worst_E_idx = -1, worst_L_idx = -1, worst_Q_idx = -1;
        FILE *fp_cons_log = fopen("conservation_errors.txt", "w");
        if (fp_cons_log) fprintf(fp_cons_log, "# photon_idx dE_rel dL_rel dQ_rel Q_init Q_final term_type final_t\\n");

        for (long int i = 0; i < num_rays; ++i) {{
            double final_E, final_Lx, final_Ly, final_Lz, final_Q;
            // UPDATED: Use dynamic function name
            {conserved_quantities_func}(commondata, all_photons[i].f, &final_E, &final_Lx, &final_Ly, &final_Lz, &final_Q);

            const double dE = (initial_conserved_quantities[i][0] != 0) ? fabs((final_E - initial_conserved_quantities[i][0]) / initial_conserved_quantities[i][0]) : fabs(final_E - initial_conserved_quantities[i][0]);
            if (dE > max_dE) {{ max_dE = dE; worst_E_idx = i; }}

            double dL;
            if (commondata->a_spin == 0.0) {{
                const double L_init_mag = sqrt(SQR(initial_conserved_quantities[i][1]) + SQR(initial_conserved_quantities[i][2]) + SQR(initial_conserved_quantities[i][3]));
                const double L_final_mag = sqrt(SQR(final_Lx) + SQR(final_Ly) + SQR(final_Lz));
                dL = (L_init_mag != 0) ? fabs((L_final_mag - L_init_mag) / L_init_mag) : fabs(L_final_mag - L_init_mag);
            }} else {{
                const double Lz_init = initial_conserved_quantities[i][3];
                dL = (Lz_init != 0) ? fabs((final_Lz - Lz_init) / Lz_init) : fabs(final_Lz - Lz_init);
            }}
            if (dL > max_dL) {{ max_dL = dL; worst_L_idx = i; }}

            const double Q_init = initial_conserved_quantities[i][4];
            const double dQ = (Q_init != 0) ? fabs((final_Q - Q_init) / Q_init) : fabs(final_Q - Q_init);
            if (dQ > max_dQ) {{ max_dQ = dQ; worst_Q_idx = i; }}

            if (fp_cons_log) fprintf(fp_cons_log, "%ld %.6e %.6e %.6e %.6e %.6e %d %.6e\\n", i, dE, dL, dQ, Q_init, final_Q, all_photons[i].status, all_photons[i].f[0]);
        }}
        if (fp_cons_log) {{ fclose(fp_cons_log); printf("Full conservation error data saved to 'conservation_errors.txt'.\\n"); }}
        printf("Max relative error in Energy (E): %.3e (photon %ld)\\n", max_dE, worst_E_idx);
        printf("Max relative error in Ang. Mom. (L): %.3e (photon %ld)\\n", max_dL, worst_L_idx);
        printf("Max relative error in Carter Constant (Q): %.3e (photon %ld)\\n", max_dQ, worst_Q_idx);
        printf("------------------------------------\\n");
    }}
    free(requests); free(metric_results); free(christoffel_results);
    if (commondata->perform_conservation_check) {{ free(initial_conserved_quantities); }}

    // === FINALIZATION ===
    printf("Processing final states and populating blueprint buffer...\\n");
    #pragma omp parallel for
    for (long int i = 0; i < num_rays; i++) {{
        results_buffer[i] = calculate_and_fill_blueprint_data_universal(
            commondata, &all_photons[i],
            window_center, n_x, n_y
        );
    }}
    free(all_photons);
    slot_manager_free(&tsm);
    """

    # Register the C function.
    cfc.register_CFunction(
        includes=includes, desc=desc, name=name, params=params, body=body
    )

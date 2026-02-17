"""
Generates the complete C project for simulating photon geodesics (ray-tracing).

This script serves as the top-level orchestrator for the updated photon geodesic 
integrator project. It generates a standalone C application that evolves photon 
trajectories in the Kerr-Schild spacetime.

Major Updates:
- Targeted specifically for "KerrSchild_Cartesian" spacetime.
- Corrected PhotonState padding to include 'double h'.
- Updated path handling for deep directory structures.
- Integrated new BHaH infrastructure and updated struct definitions.

Author: Dalton J. Moone (Updated)
"""

# ##############################################################################
# PART 0: IMPORTS AND PATH SETUP
# ##############################################################################

import os
import sys
import shutil
import argparse

# Step 0.a: Add the nrpy root directory to the Python path.
# We assume this script is located at: 
# [repo_root]/nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/
# We need to add [repo_root] to sys.path to import 'nrpy'.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 6 levels: photon -> geodesics -> gr -> bhah -> infra -> nrpy_pkg -> repo_root
nrpy_root_dir = os.path.abspath(os.path.join(script_dir, "../../../../../.."))

if nrpy_root_dir not in sys.path:
    sys.path.insert(0, nrpy_root_dir)

# Import nrpy core modules
import nrpy.params as par
import nrpy.helpers.generic as gh

# Import nrpy BHaH infrastructure modules
from nrpy.infrastructures.BHaH import (
    BHaH_defines_h,
    cmdline_input_and_parfiles,
    Makefile_helpers as Makefile,
    CodeParameters as CPs,
)

# Import Physics/Math Generators (Symbolic)
from nrpy.equations.general_relativity.geodesics import analytic_spacetimes as anasp
from nrpy.equations.general_relativity.geodesics import geodesics as geo

# Import C-Code Builder Functions
# 1. Physics Kernels
from nrpy.infrastructures.BHaH.general_relativity.geodesics import g4DD_metric
from nrpy.infrastructures.BHaH.general_relativity.geodesics import connections
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import calculate_ode_rhs
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import p0_reverse
from nrpy.infrastructures.BHaH.general_relativity.geodesics import conserved_quantities

# 2. Logic & Control
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import set_initial_conditions_cartesian
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import handle_source_plane_intersection
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import event_detection_manager
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import find_event_time_and_state
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import calculate_and_fill_blueprint_data_universal

# 3. Numerical Pipeline Helpers
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import rkf45_helpers_for_header
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import rkf45_update_and_control_helper
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import time_slot_manager_helpers
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import placeholder_interpolation_engine

# 4. Orchestrators
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import batch_integrator_numerical
from nrpy.infrastructures.BHaH.general_relativity.geodesics.photon import main


# ##############################################################################
# PART 1: CUSTOM STRUCTURE DEFINITIONS
# ##############################################################################

def register_custom_structures() -> None:
    """
    Generate C code for all custom structs and enums, then register them.
    
    Updated to correct padding logic for PhotonState and remove massive particle data.
    """
    
    consolidated_structs_c_code = r"""
// ==========================================
// Event Detection and Plane Crossing Helpers
// ==========================================
typedef struct { double n[3]; double d; } plane_event_params;

// Function pointer type for generic event functions (e.g. plane distance)
typedef double (*event_function_t)(const double f[9], void *event_params);

static inline double plane_event_func(const double f[9], void *event_params) {
    plane_event_params *params = (plane_event_params *)event_params;
    return f[1]*params->n[0] + f[2]*params->n[1] + f[3]*params->n[2] - params->d;
}

// ==========================================
// Batch Integration and Output Structs
// ==========================================

// Request struct: ID + 4D position (t, x, y, z)
typedef struct { int photon_id; double pos[4]; } photon_request_t;

// Event data struct: Results of a crossing finding
typedef struct { 
    bool found; 
    double lambda_event; 
    double t_event; 
    double f_event[9]; 
} event_data_struct;

// Termination status enum
typedef enum {
    FAILURE_PT_TOO_BIG, 
    FAILURE_RKF45_REJECTION_LIMIT, 
    FAILURE_T_MAX_EXCEEDED,
    FAILURE_SLOT_MANAGER_ERROR, 
    TERMINATION_TYPE_FAILURE,
    TERMINATION_TYPE_SOURCE_PLANE,
    TERMINATION_TYPE_CELESTIAL_SPHERE, 
    ACTIVE,
} termination_type_t;

// Final output blueprint struct (Packed for binary output)
typedef struct {
    termination_type_t termination_type; 
    double y_w, z_w; 
    double y_s, z_s; 
    double final_theta, final_phi; 
    double L_w, t_w, L_s, t_s;
} __attribute__((packed)) blueprint_data_t;

// ==========================================
// Main Photon State Struct
// ==========================================
#define CACHE_LINE_SIZE 64
#define BUNDLE_CAPACITY 16384

typedef struct {
    // 3 states for interpolation: Current, Previous, Pre-Previous
    // 9 * 3 = 27 doubles
    double f[9], f_p[9], f_p_p[9];
    
    // Affine parameter history
    // 3 doubles
    double affine_param, affine_param_p, affine_param_p_p;
    
    // Current step size
    // 1 double
    double h; 
    
    termination_type_t status;
    int rejection_retries;
    
    // Crossing flags
    bool on_positive_side_of_window_prev; 
    bool on_positive_side_of_source_prev;
    
    // Event data storage
    event_data_struct source_event_data;
    event_data_struct window_event_data;
    
    // Padding to ensure 64-byte alignment/size
    // Calculation Updates:
    // double arrays (3*9 + 3 + 1(h)) * 8 = 31 * 8 = 248 bytes
    // status (4) + retries (4) = 8 bytes
    // bools (2) = 2 bytes
    // event_structs: size depends on struct packing, handled by sizeof()
    char _padding[CACHE_LINE_SIZE - (
        sizeof(double)*31 + 
        sizeof(termination_type_t) + 
        sizeof(int) +
        sizeof(bool)*2 + 
        sizeof(event_data_struct)*2
    ) % CACHE_LINE_SIZE];
} __attribute__((aligned(CACHE_LINE_SIZE))) PhotonState;
"""
    BHaH_defines_h.register_BHaH_defines("after_general", consolidated_structs_c_code)


# ##############################################################################
# PART 2: MAIN CONFIGURATION
# ##############################################################################

if __name__ == "__main__":
    # Step 1: Set up arguments
    parser = argparse.ArgumentParser(
        description="Generate the Updated Photon Geodesic Integrator (Kerr-Schild)."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="project",
        help="The parent directory where the C project will be generated.",
    )
    args = parser.parse_args()

    # Step 2: Define Project Constants
    project_name = "photon_geodesic_integrator"
    exec_name = "photon_geodesic_integrator"
    project_dir = os.path.join(args.outdir, project_name)
    
    # Configuration
    SPACETIME = "KerrSchild_Cartesian"
    PARTICLE = "photon"
    GEO_KEY = f"{SPACETIME}_{PARTICLE}"

    # Step 3: Setup Directory and Core Infrastructure
    print(f"Initializing project: {project_name}")
    shutil.rmtree(project_dir, ignore_errors=True)
    os.makedirs(project_dir, exist_ok=True)
    
    # Set global NRPy parameter
    par.set_parval_from_str("Infrastructure", "BHaH")

    # Step 4: Register CodeParameters
    print(" -> Registering CodeParameters...")
    param_defs = {
        "REAL": [
            ("M_scale", 1.0), 
            ("a_spin", 0.9),
            
            # Camera / Window Geometry
            ("camera_pos_x", 0.0), ("camera_pos_y", 0.0), ("camera_pos_z", 51.0),
            ("window_center_x", 0.0), ("window_center_y", 0.0), ("window_center_z", 50.0),
            ("window_up_vec_x", 0.0), ("window_up_vec_y", 1.0), ("window_up_vec_z", 0.0),
            ("window_size", 1.5),
            
            # Source Plane Geometry (The "Floor")
            ("source_plane_normal_x", 0.0), ("source_plane_normal_y", 0.0), ("source_plane_normal_z", 1.0),
            ("source_plane_center_x", 0.0), ("source_plane_center_y", 0.0), ("source_plane_center_z", 0.0),
            ("source_up_vec_x", 0.0), ("source_up_vec_y", 1.0), ("source_up_vec_z", 0.0),
            ("source_r_min", 6.0), ("source_r_max", 25.0),
            
            # Integration Control
            ("t_start", 2000.0), 
            ("t_integration_max", 10000.0), 
            ("r_escape", 1500.0), 
            ("p_t_max", 1000.0),
            ("slot_manager_t_min", -100.0), 
            ("slot_manager_delta_t", 0.1),
            ("numerical_initial_h", 1.0), 
            
            # RKF45 Tolerances
            ("rkf45_error_tolerance", 1e-8),
            ("rkf45_absolute_error_tolerance", 1e-8), 
            ("rkf45_h_min", 1e-10),
            ("rkf45_h_max", 10.0), 
            ("rkf45_safety_factor", 0.9),
        ],
        "int": [
            ("scan_density", 100), 
            ("rkf45_max_retries", 10),
        ],
        "bool": [
            ("perform_conservation_check", True), 
            ("debug_mode", False),
        ],
    }

    for c_type, params_list in param_defs.items():
        for name, default in params_list:
            par.register_CodeParameter(
                c_type, __name__, name, default, commondata=True, add_to_parfile=True
            )

    # Step 5: Acquire Symbolic Physics Expressions
    print(f" -> Acquiring symbolic data for {GEO_KEY}...")
    metric_data = anasp.Analytic_Spacetimes[SPACETIME]
    geodesic_data = geo.Geodesic_Equations[GEO_KEY]

    # Step 6: Generate C Code for Parameters (Header & Parser)
    print(" -> Generating parameter handling code...")
    CPs.write_CodeParameters_h_files(project_dir=project_dir, set_commondata_only=True)
    CPs.register_CFunctions_params_commondata_struct_set_to_default()
    cmdline_input_and_parfiles.generate_default_parfile(
        project_dir=project_dir, project_name=project_name
    )
    cmdline_inputs_list = [name for _, params in param_defs.items() for name, _ in params]
    cmdline_input_and_parfiles.register_CFunction_cmdline_input_and_parfile_parser(
        project_name=project_name, cmdline_inputs=cmdline_inputs_list
    )

    # Step 7: Register all C Functions
    print(" -> Registering C functions...")

    # A. Physics Kernels
    # Note: These generate the Metric and Connection structs automatically now.
    g4DD_metric.g4DD_metric(metric_data.g4DD, SPACETIME, PARTICLE)
    connections.connections(geodesic_data.Gamma4UDD, SPACETIME, PARTICLE)
    
    # ODE RHS (Equations of Motion)
    # Uses f[9] state vector: [t, x, y, z, pt, px, py, pz, L]
    calculate_ode_rhs.calculate_ode_rhs(geodesic_data.geodesic_rhs, geodesic_data.xx)
    
    # Initial Condition Constraints
    if geodesic_data.p0_photon is None:
        raise ValueError(f"p0_photon symbolic expression is missing for {GEO_KEY}")
    p0_reverse.p0_reverse(geodesic_data.p0_photon)
    
    # Diagnostics
    conserved_quantities.conserved_quantities(SPACETIME, PARTICLE)

    # B. Initialization and Setup
    set_initial_conditions_cartesian.set_initial_conditions_cartesian(SPACETIME)
    
    # C. Event Detection & Handling
    event_detection_manager.event_detection_manager()
    find_event_time_and_state.find_event_time_and_state() # Register the interpolation engine
    handle_source_plane_intersection.handle_source_plane_intersection()
    calculate_and_fill_blueprint_data_universal.calculate_and_fill_blueprint_data_universal()

    # D. Numerical Integration Infrastructure
    # These inject static inline helpers into BHaH_defines.h
    rkf45_helpers_for_header.rkf45_helpers_for_header(SPACETIME)
    rkf45_update_and_control_helper.rkf45_update_and_control_helper()
    time_slot_manager_helpers.time_slot_manager_helpers()
    
    # Placeholder engine for batch interpolation (calls analytic metric directly)
    placeholder_interpolation_engine.placeholder_interpolation_engine(SPACETIME, PARTICLE)

    # E. Orchestrators
    # The batch integrator drives the numerical loop
    batch_integrator_numerical.batch_integrator_numerical(SPACETIME)
    
    # The main() function
    main.main(SPACETIME)

    # Step 8: Assemble Final C Project
    print(" -> Assembling C project on disk...")
    
    # A. Register the custom structs we defined at the top
    register_custom_structures()
    
    # B. Output BHaH_defines.h (includes all registered structs and inline helpers)
    BHaH_defines_h.output_BHaH_defines_h(project_dir=project_dir, enable_rfm_precompute=False)
    
    # C. Copy intrinsics (standard nrpy requirement)
    gh.copy_files(
        package="nrpy.helpers",
        filenames_list=["simd_intrinsics.h"],
        project_dir=project_dir,
        subdirectory="intrinsics",
    )
    
    # D. Generate Makefile
    # We add -fopenmp for the parallel batch integrator
    Makefile.output_CFunctions_function_prototypes_and_construct_Makefile(
        project_dir=project_dir,
        project_name=project_name,
        exec_or_library_name=exec_name,
        addl_CFLAGS=["-Wall -Wextra -g -fopenmp -O3"],
        addl_libraries=["-lm -fopenmp"], 
    )

    print("----------------------------------------------------------")
    print(f"Finished! C project generated in: {project_dir}/")
    print("To compile and run:")
    print(f"  cd {project_dir}")
    print("  make")
    print(f"  ./{exec_name}")
    print("----------------------------------------------------------")
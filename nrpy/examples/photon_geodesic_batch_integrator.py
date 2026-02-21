"""
Generates the complete C project for simulating photon geodesics (ray-tracing).

This script serves as the top-level orchestrator for the updated photon geodesic 
integrator project. It generates a standalone C application that evolves photon 
trajectories in the Kerr-Schild spacetime.

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
# PART 1: MAIN CONFIGURATION
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

    # Step 4: Acquire Symbolic Physics Expressions
    print(f" -> Acquiring symbolic data for {GEO_KEY}...")
    metric_data = anasp.Analytic_Spacetimes[SPACETIME]
    geodesic_data = geo.Geodesic_Equations[GEO_KEY]

    # Step 5: Execute Modules and Register C Functions
    # CRITICAL: This must happen BEFORE Step 6 so that all distributed 
    # CodeParameters are registered into NRPy's global dictionary first.
    print(" -> Registering C functions and local CodeParameters...")

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

    # Step 6: Generate C Code for Parameters (Header & Parser)
    # Now that all modules have run, NRPy knows about all the parameters.
    print(" -> Generating parameter handling code...")
    CPs.write_CodeParameters_h_files(project_dir=project_dir, set_commondata_only=True)
    CPs.register_CFunctions_params_commondata_struct_set_to_default()
    cmdline_input_and_parfiles.generate_default_parfile(
        project_dir=project_dir, project_name=project_name
    )
    
    # Dynamically gather all registered parameters for the command-line parser
    cmdline_inputs_list = list(par.glb_code_params_dict.keys())
    cmdline_input_and_parfiles.register_CFunction_cmdline_input_and_parfile_parser(
        project_name=project_name, cmdline_inputs=cmdline_inputs_list
    )

    # Step 7: Assemble Final C Project
    print(" -> Assembling C project on disk...")
    
    # A. Output BHaH_defines.h (includes all registered structs and inline helpers)
    BHaH_defines_h.output_BHaH_defines_h(project_dir=project_dir, enable_rfm_precompute=False)
    
    # B. Copy intrinsics (standard nrpy requirement)
    gh.copy_files(
        package="nrpy.helpers",
        filenames_list=["simd_intrinsics.h"],
        project_dir=project_dir,
        subdirectory="intrinsics",
    )
    
    # C. Generate Makefile
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
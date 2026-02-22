import numpy as np
import os

# --- 1. Core Data Structures ---
BLUEPRINT_DTYPE = np.dtype([
    ('termination_type', np.int32),
    ('y_w', 'f8'),
    ('z_w', 'f8'),
    ('y_s', 'f8'),
    ('z_s', 'f8'),
    ('final_theta', 'f8'),
    ('final_phi', 'f8'),
    ('L_w', 'f8'),
    ('t_w', 'f8'),
    ('L_s', 'f8'),
    ('t_s', 'f8'),
], align=False)

# --- 2. Termination Enums ---
# Updated to match the C code's removal of the WINDOW_PLANE enum
TERM_SPHERE = 0
TERM_SOURCE_PLANE = 1
TERM_FAIL_PT_BIG = 2
TERM_FAIL_RKF45 = 3
TERM_FAIL_T_MAX = 4
TERM_FAIL_SLOT = 5
TERM_FAIL_GENERIC = 6
TERM_ACTIVE = 7

# --- 3. Global Directories ---
HOME_DIR = os.path.expanduser('~')
BASE_PROJECT_DIR = os.path.join(HOME_DIR, "Desktop", "Test_PR", "nrpy", "project")
LIGHT_INTEGRATOR_DIR = os.path.join(BASE_PROJECT_DIR, "photon_geodesic_integrator")
OUTPUT_BASEDIR = os.path.join(HOME_DIR, "Desktop", "Test_PR", "nrpy", "Generated_nrpy_images")

# --- 4. Physics & Scene Parameters ---
MASS_OF_BLACK_HOLE = 1.0
WINDOW_WIDTH = 1.0

# --- 5. Texture & Disk Generation Parameters ---
SPHERE_TEXTURE_FILE = "starmap_2020.png"
DISK_INNER_RADIUS = 6.0
DISK_OUTER_RADIUS = 25.0
COLORMAP = 'afmhot'
DISK_TEMP_POWER_LAW = -1.5
SOURCE_PHYSICAL_WIDTH = 2 * DISK_OUTER_RADIUS

# --- 6. Rendering Parameters ---
STATIC_IMAGE_PIXEL_WIDTH = 700
CHUNK_SIZE = 10_000_000
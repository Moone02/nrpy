import numpy as np
import os
import sys

# Temporarily add the script's directory to sys.path so it can find config_and_types.py
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    import config_and_types as cfg
except ImportError:
    print(f"[!] ERROR: Cannot find 'config_and_types.py' in {script_dir}")
    print("    Please ensure it is saved in the same directory as this script.")
    sys.exit(1)

def diagnose_blueprint(blueprint_path: str = None) -> None:
    """
    Reads the binary blueprint file and prints critical diagnostics to help 
    track down rendering issues (like pure black images).
    """
    if blueprint_path is None:
        # Dynamically resolve the path to the project directory
        # Up 3 levels from nrpy/nrpy/helpers/geodesic_visualizations to nrpy/
        blueprint_path = os.path.abspath(os.path.join(
            script_dir, "..", "..", "..", "project", "photon_geodesic_integrator", "light_blueprint.bin"
        ))
    
    print("=================================================================")
    print(f" BLUEPRINT DIAGNOSTICS")
    print(f" File: {blueprint_path}")
    print("=================================================================")
    
    if not os.path.exists(blueprint_path):
        print(f"[!] ERROR: Blueprint file not found at:\n    {blueprint_path}")
        return

    # Load data using our struct definition
    data = np.fromfile(blueprint_path, dtype=cfg.BLUEPRINT_DTYPE)
    total_rays = len(data)
    print(f"Total records loaded: {total_rays:,}\n")

    # --- 1. Check Enum Mappings ---
    print("--- 1. Raw Termination Enums in Binary ---")
    unique_enums, counts = np.unique(data['termination_type'], return_counts=True)
    
    for e, c in zip(unique_enums, counts):
        print(f"  Raw Enum {e:2d}: {c:8,} rays ({c/total_rays*100:6.2f}%)")
        
    print("\n  [Config Expected]: SOURCE_PLANE = 2, SPHERE = 3")
    print("  -> If your raw enums above do NOT match 2 and 3, your image will ")
    print("     render completely black. You must update config_and_types.py ")
    print("     to match the actual C enum integer values.")

    # --- 2. Check Window Coordinates (Field of View) ---
    print("\n--- 2. Window Coordinate Bounds (y_w, z_w) ---")
    valid_yw = data['y_w'][np.isfinite(data['y_w'])]
    valid_zw = data['z_w'][np.isfinite(data['z_w'])]
    
    if len(valid_yw) > 0:
        min_yw, max_yw = np.min(valid_yw), np.max(valid_yw)
        min_zw, max_zw = np.min(valid_zw), np.max(valid_zw)
        
        print(f"  Calculated y_w bounds: [{min_yw:8.4f}, {max_yw:8.4f}]")
        print(f"  Calculated z_w bounds: [{min_zw:8.4f}, {max_zw:8.4f}]")
        
        half_w = cfg.WINDOW_WIDTH / 2.0
        print(f"  Renderer FOV limit:    [{-half_w:8.4f}, {half_w:8.4f}] (cfg.WINDOW_WIDTH = {cfg.WINDOW_WIDTH})")
        
        # Count how many rays actually hit inside the camera's FOV crop
        in_view = np.sum(
            (data['y_w'] >= -half_w) & (data['y_w'] < half_w) & 
            (data['z_w'] >= -half_w) & (data['z_w'] < half_w)
        )
        print(f"\n  Rays inside Renderer FOV: {in_view:,} ({in_view/total_rays*100:.2f}%)")
        print("  -> If this number is 0, your camera window width is too small ")
        print("     or the initial conditions mapped the rays incorrectly.")
    else:
        print("  [!] ERROR: All y_w and z_w values are NaN or Infinity!")

    # --- 3. View Raw Sample Data ---
    print("\n--- 3. First 5 Raw Records ---")
    header = f"{'Ray#':<6} | {'Term Type':<9} | {'y_w':>8} | {'z_w':>8} | {'y_s / theta':>11} | {'z_s / phi':>11}"
    print(header)
    print("-" * len(header))
    
    for i in range(min(5, total_rays)):
        rec = data[i]
        tt = rec['termination_type']
        print(f"{i:<6} | {tt:<9} | {rec['y_w']:>8.2f} | {rec['z_w']:>8.2f} | "
              f"{rec['y_s'] if tt != cfg.TERM_SPHERE else rec['final_theta']:>11.3f} | "
              f"{rec['z_s'] if tt != cfg.TERM_SPHERE else rec['final_phi']:>11.3f}")
    
    print("\n=================================================================")

if __name__ == "__main__":
    diagnose_blueprint()

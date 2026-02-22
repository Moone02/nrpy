import struct
import os
from collections import Counter

# Termination types from batch_integrator_numerical.py
TERMINATION_ENUM = [
    "FAILURE_PT_TOO_BIG",
    "FAILURE_RKF45_REJECTION_LIMIT",
    "FAILURE_T_MAX_EXCEEDED",
    "FAILURE_SLOT_MANAGER_ERROR",
    "TERMINATION_TYPE_FAILURE",
    "TERMINATION_TYPE_SOURCE_PLANE",
    "TERMINATION_TYPE_WINDOW_PLANE",
    "TERMINATION_TYPE_CELESTIAL_SPHERE",
    "ACTIVE"
]

def analyze_blueprint(filename="light_blueprint.bin"):
    # Determine the correct path if run from the top address
    if not os.path.exists(filename):
        alt_path = os.path.join("project", "photon_geodesic_integrator", filename)
        if os.path.exists(alt_path):
            filename = alt_path
        else:
            print(f"Error: Could not find '{filename}' in current directory or project path.")
            return

    # blueprint_data_t: 1 int (4 bytes) + 10 doubles (80 bytes) = 84 bytes
    # '=' enforces standard sizes (no padding) consistent with __attribute__((packed))
    fmt = "=i10d"
    record_size = struct.calcsize(fmt)
    
    stats = Counter()
    crossed_window = 0
    total = 0

    try:
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(record_size)
                if not chunk:
                    break
                if len(chunk) < record_size:
                    print(f"Warning: Trailing data detected ({len(chunk)} bytes).")
                    break
                
                # Unpack: (status, y_w, z_w, y_s, z_s, theta, phi, L_w, t_w, L_s, t_s)
                data = struct.unpack(fmt, chunk)
                status_idx = data[0]
                L_w = data[7] # Distance L at window intersection
                
                stats[status_idx] += 1
                if L_w != 0.0:
                    crossed_window += 1
                total += 1
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if total == 0:
        print("No records found.")
        return

    print("\n" + "="*65)
    print(f" RAY TRACING RESULTS ANALYSIS: {filename}")
    print("="*65)
    print(f"{'TERMINATION STATUS':<35} | {'COUNT':>8} | {'PERCENT':>10}")
    print("-" * 65)
    
    for i, name in enumerate(TERMINATION_ENUM):
        count = stats.get(i, 0)
        percentage = (count / total) * 100
        print(f"{name:<35} | {count:8} | {percentage:9.2f}%")
        
    print("-" * 65)
    print(f"{'Total Photons':<35} | {total:8} | 100.00%")
    print("-" * 65)
    
    win_pct = (crossed_window / total) * 100
    print(f"Photons successfully crossing Window Plane:  {crossed_window:8} ({win_pct:6.2f}%)")
    print(f"Photons that NEVER crossed the Window:      {total - crossed_window:8} ({100-win_pct:6.2f}%)")
    print("="*65 + "\n")

if __name__ == "__main__":
    analyze_blueprint()
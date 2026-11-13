# Geodesic Raytracing

> Explain standalone massive and photon geodesic examples plus batch photon raytracing visualization files. · Status: confirmed
> Up: [Examples](index.md)

## Summary

NRPy has checked-in geodesic example generators. `massive_single_geodesic_integrator_analytical`
builds a single massive-particle Kerr-Schild Cartesian trajectory and uses GSL's
RKF45 ODE path. `photon_single_geodesic_integrator_analytical` builds a single photon trajectory
from either the Kerr-Schild recipe or the static coincident Brill-Lindquist
recipe and uses the split photon RKF45 kernels directly.
`photon_batch_geodesic_integrator_analytical` builds a tiled photon
raytracing project, defaults to OpenMP, can generate CUDA code with `--cuda`,
honors `--outdir`, and writes per-tile light-blueprint binary files for the
lensed-image renderer and diagnostic scripts. The numerical single- and batch-
photon generators use the same photon runtime with a combined numerical-
spacetime `.bin` dataset and currently support CPU/OpenMP
`SinhCylindricalv2n2` interpolation.

All four photon generators initialize rays with a metric-orthonormal camera
tetrad and unit initial camera-tetrad energy magnitude, `E_camera = 1`. This
affine normalization differs from the hypersurface-normal measure
`|alpha p^0|`. Direct and normalized photon EOMs use the corresponding
upper-only log-energy termination measure: direct states evaluate
`ln|alpha p^0|`, while normalized states use `u`.

The commands on this page describe manual generation, compilation, trajectory
integration, and rendering. Neither GitHub workflow nor the local full-CI
helper invokes the geodesic generators described here.

## Detail

From a repository checkout, run generators from the repository root. If NRPy is
not installed editable, first set `PYTHONPATH` as described by
[Build And Run](../architecture/build-and-run.md):

```bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
```

For a massive single-ray run:

```bash
python -m nrpy.examples.massive_single_geodesic_integrator_analytical
cd project/massive_single_geodesic_integrator_analytical
make
./massive_single_geodesic_integrator_analytical
python3 visualize_trajectory.py --particle_type Massive
```

The generated C program initializes one massive test particle in
`KerrSchild_Cartesian`, computes `u^t` from the massive normalization
constraint, integrates an eight-component state through `gsl_odeiv2_step_rkf45`,
writes `trajectory.txt`, then reports final normalization and conserved-quantity
errors. Its generated Makefile uses `$(shell gsl-config --cflags)` and
`$(shell gsl-config --libs)`, so GSL is a build dependency sourced from the
generator. Official GSL [Using the Library](https://www.gnu.org/software/gsl/doc/html/usage.html)
headings `Compiling and Linking` and `Linking programs with the library`
describe the external header/linker prerequisite; exact `gsl-config` use is
NRPy generator behavior.

For a photon single-ray run:

```bash
python -m nrpy.examples.photon_single_geodesic_integrator_analytical \
  --observer-position 0.0 -10.0 0.0 \
  --observer-look-forward 0.0 1.0 0.0 \
  --observer-up 0.0 0.0 1.0 \
  --escape-radius 25.0
cd project/photon_single_geodesic_integrator_analytical
make
./photon_single_geodesic_integrator_analytical
python3 visualize_trajectory.py --particle_type Photon
```

The analytical photon commands evaluate metric and Christoffel components at
each fixed-anchor RKF45 Brent trial. The event manager records accepted
crossing steps before physical stopping checks.

The analytical single-photon path defaults to `KerrSchild_Cartesian` and also
accepts `BrillLindquist_InitialData_Static_Cartesian` through `--spacetime`.
It accepts direct or normalized photon equations through `--eom` and does not
use GSL. It allocates one Structure-of-Arrays photon state, constructs the complete
initial momentum from the observer tetrad, and evaluates the split RKF45 stages through
`interpolation_kernel`, `calculate_ode_rhs_kernel`, `rkf45_stage_update`, and
`rkf45_finalize_and_control`, writes `trajectory.txt`, and reports null
normalization and conserved-quantity errors. Accepted states also use the
normal-observer log-energy cutoff, whose default `--evolution-measure-max` is
`3` and which triggers only when the log measure is above the threshold. Both single-ray generators copy
`visualize_trajectory.py` into the generated project and print
`pip install matplotlib numpy`; those Python visualization dependencies are
source-limited to the checked-in script imports and generator message.
Both single-photon generators accept optional plane settings. Supply every
option in each enabled group:

- Terminal: `--terminal-plane-center`, `--terminal-plane-normal`,
  `--terminal-plane-up`, and `--terminal-plane-radius`.
- Nonterminal: `--non-terminal-plane-center`, `--non-terminal-plane-normal`,
  and `--non-terminal-plane-up`.

Both single-photon generators accept
`--plane-crossing-residual-tolerance` and
`--plane-crossing-parameter-tolerance`, each defaulting to `1e-12`. After an
accepted RKF45 step changes sides of a plane, it saves the state before that
step and both integration parameters. The photon continues evolving. After
evolution, each Brent trial takes one fifth-order RKF45 step from the saved
state. The numerical integrator selects the first terminal crossing inside the
radial bounds and keeps a nonterminal crossing only when it precedes that hit.

Both single-photon integrators write coordinate time, affine parameter,
local plane coordinates, nine state components, `brent_iterations`, and
`signed_plane_distance` to `plane_crossings.txt`. For direct EOM, state
components are `t`, `x`, `y`, `z`, `p^t`, `p^x`, `p^y`, `p^z`, and `L_normal`.
For normalized EOM, they are `lambda`, `x`, `y`, `z`, `u`, `Pi_1`, `Pi_2`,
`Pi_3`, and `L_normal`.

The numerical single-photon executable also writes `initial_state.txt` with
the observer event and initial direct `p^mu` at full precision. For normalized
evolution, it records that momentum before converting to `u` and `Pi_i`.

For numerical spacetime, an in-range terminal crossing sets final status after
trajectory integration. A nonterminal crossing does not stop integration.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py` and `single_integrator_numerical.py` — `plane_crossings.txt` output and EOM state columns; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py` — accepted-step crossing capture; `find_plane_crossings_brent.py` — analytical and numerical refinement; both single-photon generators — Brent-tolerance CLI

Both single-photon generators enable RKF45 trial and stage diagnostics by
default. Their generated executables write adaptive-step controller values to
`rkf45_trials.txt` and stage evaluations to `rkf45_stages.txt`. Analytical runs
record all six metric, connection, and right-hand-side evaluations per trial;
numerical runs record only successfully interpolated stages. Pass
`--disable-rkf45-trial-debug` during project generation to omit this diagnostic
code; use `--enable-rkf45-trial-debug` to select it explicitly.

Evidence (descriptive behavior): `nrpy/examples/photon_single_geodesic_integrator_analytical.py` and `photon_single_geodesic_integrator_numerical.py` — plane options and debug defaults; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py` and `single_integrator_numerical.py` — generated C behavior; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py` — crossing and termination checks

`trajectory.txt` is the input file for the single-ray visualization. The
massive file header is `# proper_time t x y z u^t u^x u^y u^z`. Direct photon
output uses `# lambda t x y z p^t p^x p^y p^z L_normal`; normalized output uses
`# lambda t x y z u Pi_1 Pi_2 Pi_3 L_normal`. Numerical photon output appends a
`norm` column containing the accepted-state constraint residual.
`visualize_trajectory.py` loads the text file with NumPy, validates it is present
and non-empty, prints initial and final spatial positions plus the accumulated
affine or proper parameter, then saves `trajectory_plot.png` and displays a
Matplotlib 3D path with a schematic `r=2M` reference sphere.

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/visualize_trajectory.py` — `visualize_trajectory`; `nrpy/examples/photon_single_geodesic_integrator_analytical.py` — trajectory header; corroboration: `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — numerical trajectory header

For a batch lensed-image run with default OpenMP output under `project/`:

```bash
python -m nrpy.examples.photon_batch_geodesic_integrator_analytical \
  --observer-position 0.0 -10.0 0.0 \
  --observer-look-forward 0.0 1.0 0.0 \
  --observer-up 0.0 0.0 1.0 \
  --observer-fov 1.0 1.0 \
  --scan-density 500 \
  --escape-radius 25.0
cd project/photon_batch_geodesic_integrator_analytical
make
./photon_batch_geodesic_integrator_analytical
python3 visualize_lensed_image.py --terminal-plane-radius 6.0 20.0 --pixel-width 600
python3 blueprint_analysis.py
```

For a CUDA batch project or a different parent output directory:

```bash
python -m nrpy.examples.photon_batch_geodesic_integrator_analytical \
  --cuda --outdir /tmp/nrpy-geodesics \
  --observer-position 0.0 -10.0 0.0 \
  --observer-look-forward 0.0 1.0 0.0 \
  --observer-up 0.0 0.0 1.0 \
  --observer-fov 1.0 1.0 \
  --scan-density 500 \
  --escape-radius 25.0
cd /tmp/nrpy-geodesics/photon_batch_geodesic_integrator_analytical
make
./photon_batch_geodesic_integrator_analytical
```

The analytical batch integrator saves crossing steps and transfers CUDA
crossing records to the host before Brent refinement. Analytical CUDA event
capture runs on the device; terminal and nonterminal projections run on the
host after transfer. Analytical event capture has no numerical time-window stop.

The batch generator derives `project_dir` from `--outdir` plus
`photon_batch_geodesic_integrator_analytical`. It sets `parallelization` to `openmp` unless
`--cuda` is present. OpenMP uses `gcc`, `-fopenmp`, and C sources; CUDA uses
`nvcc`, `-lcudart`, `-DUSE_GPU`, `.cu` sources, and copied `cuda_intrinsics.h`.
Both modes require `--scan-density` for the number of photon columns per tile,
and `--tile-counts` defaults to `1 1`. For a full-image scan without range
options, `main_batch.main` calculates rows per tile from the projected camera
height-to-width ratio and the tile-count ratio, using nearest-integer rounding
with `llround`. Generated programs expose `tiles_width`, `tiles_height`, and
`scan_density` as photon-sampling parameters. The renderer's `--pixel-width`
sets output resolution independently of the photon count; it reads tile counts,
full-camera fields of view, and exact scan bounds from blueprint headers.
The CUDA runtime/toolchain facts are limited to
the generator's compiler, flags, and copied helper declarations. NVIDIA's
official [NVCC guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
heading `NVCC: The NVIDIA CUDA Compiler` establishes that `nvcc` belongs to the
CUDA Toolkit; this page does not claim any GPU model/toolkit version was tested.

Evidence (public/scientific contract): `nrpy/examples/photon_batch_geodesic_integrator_analytical.py` — argument parser and project generation; corroboration: `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — header-driven tile and aspect-ratio handling

The analytical and numerical photon examples, for both batches and single
photons, accept optional `--a-range A_MIN A_MAX` and `--b-range B_MIN B_MAX`.
These dimensionless coordinates select a rectangle of the original camera
image: `a` increases horizontally and `b` increases upward. Each omitted range
is `[0,1]`. Bounds must be finite and satisfy `0 <= MIN < MAX <= 1`.
Supplying either option, including an explicit `[0,1]` range, enables cropping.
Cropped batch scans require `--tile-counts 1 1`; incompatible options stop
before C generation. The generated batch program and shared initializer also
reject cropped scans with multiple tiles when parameters are overridden.

For a cropped batch, let $N_a$ be `scan_density`,
$\Delta a=a_{\max}-a_{\min}$, and $\Delta b=b_{\max}-b_{\min}$. The rows are

$$
N_b=\left\lfloor N_a\frac{\Delta b}{\Delta a}
\frac{\tan(\alpha_h/2)}{\tan(\alpha_w/2)}\right\rfloor.
$$

The selected width still receives $N_a$ photons per row. An empty derived row
count is an error. The sample centers are
$a_i=a_{\min}+\Delta a(i+1/2)/N_a$ and
$b_j=b_{\min}+\Delta b(j+1/2)/N_b$. The full-camera fields of view and observer
tetrad retain their meanings; cropping selects directions within that camera.
Without either range option, the existing tile-aware calculation is
$N_b=\operatorname{llround}[N_a\tan(\alpha_h/2)T_w/
(\tan(\alpha_w/2)T_h)]$, where $T_w,T_h$ are the tile counts. Each tile has
$N_aN_b$ photons.

The single-photon integrators pass one column and one row to the shared
initializer, selecting the cropped rectangle's center independently of its
aspect ratio. The numerical single-photon program retains runtime tile
selection for uncropped rays. Its cropped rays require one tile. To compare a
single photon with a batch sample, match the metric parameters, observer event,
fields of view, and sample-center coordinates; the analytical examples have
different default Kerr spins.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py` — `main`; `set_initial_conditions_kernel.py` — `set_initial_conditions_kernel`; corroboration: `nrpy/examples/photon_batch_geodesic_integrator_analytical.py`, `photon_batch_geodesic_integrator_numerical.py`, `photon_single_geodesic_integrator_analytical.py`, and `photon_single_geodesic_integrator_numerical.py` — range options and parameter defaults; `single_integrator_analytical.py` and `single_integrator_numerical.py` — one-row initialization

The analytical batch integrator applies the same upper-only log-energy cutoff
after each accepted state. Its direct-EOM path refreshes the accepted-state
metric before computing `ln|alpha p^0|`; normalized mode uses `u` directly.

The batch executable produces tiled `light_blueprint_XX_YY.bin` files in its
project directory. Those binary files are generated output, not KB sources.
The native same-build blueprint binary layout is version 9 with a 92-byte header
and 124-byte photon records. The header stores tile identity/counts,
`alpha_w`/`alpha_h`, and the exact global image-plane scan bounds
`a_min`, `a_max`, `b_min`, and `b_max`. Full-image scans store `[0,1]` in both
directions. The renderer reads these bounds automatically; `--pixel-width` spans
the stored `a` range. For requested pixel width $W$, the output height is

$$
H=\max\left(1,\left\lfloor W\frac{\Delta b}{\Delta a}
\frac{\tan(\alpha_h/2)}{\tan(\alpha_w/2)}\right\rfloor\right).
$$

Floor rounding avoids adding an unsampled row when rendering a cropped scan
at its sampling width. Output resolution does not add photons. Range arguments
are not needed when regenerating an image. Retain the texture files and the
rendering command, including pixel width, terminal-plane texture radii, and any
debug-color options, to reproduce its appearance. Readers require the current
binary-layout version; older blueprints must be regenerated with the matching
photon project and reader.

Each photon record stores
nonterminal-plane crossing coordinates, affine parameter, and coordinate time;
terminal-plane texture coordinates; final sphere angles, affine parameter, and
coordinate time; and normalized `image_width_fraction`/`image_height_fraction`
sample coordinates. It also stores axial angular momentum, coordinate time, and
signed plane distance at the Brent-refined nonterminal crossing.
The Python `BLUEPRINT_DTYPE`, C `blueprint_data_t`, and termination enums must
agree. Photon record coordinates remain global `(a,b)` values; the renderer
subtracts the stored minima and divides by the stored range widths before
placing them in output pixels.

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — `BLUEPRINT_SCHEMA_VERSION`, `BLUEPRINT_DTYPE`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/calculate_and_fill_blueprint_data_universal.py` — generated record layout

`visualize_lensed_image.py` expects those per-tile `.bin` files next to the
script and discovers the tile grid and angular aspect ratio from headers. It
does not use plane coordinates to place pixels: the renderer maps each record's
normalized image fractions with floor-to-pixel placement, preserves the
observer's horizontal orientation, and applies the required vertical
raster-row flip. It
downloads `noirlab2430b.tif` at runtime from the URL embedded in the script if
the texture is missing, generates an accretion-disk texture, and calls
`render_lensed_image.generate_static_lensed_image`. The renderer streams binary
records with `BLUEPRINT_DTYPE`, maps terminal-plane hits to the disk
texture and sphere escapes to the starmap, processes tiles in a bounded process
pool, and writes the final PNG. `render_lensed_image.py` has a no-op `numba`
fallback if import fails, but the batch generator still prints
`pip install matplotlib numpy numba Pillow`; that dependency statement is
source-limited to the checked-in files.

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — `main`; `render_lensed_image.py` — `generate_static_lensed_image`; corroboration: `nrpy/examples/geodesic_visualizations/blueprint_io.py` — binary header and record readers

`blueprint_analysis.py` is the diagnostic script for the same binary files.
It streams each tile, counts raw termination enums, compares them to
`blueprint_config_and_schema.py`, reports nonterminal-plane diagnostic statistics,
prints early records, and displays heatmaps for nonterminal-plane, terminal-plane,
and celestial-sphere coordinates. When every matching
`light_blueprint_norm_abs_XX_YY.bin` sidecar is present and correctly sized, it
also plots normalization-magnitude histograms grouped by termination status.
It also assigns every photon to a domain of equal termination status using
eight-neighbor connectivity, including diagonals and neighbors across tile
edges. Domain IDs follow increasing seed `a`, then `b`. The complete status and
domain-ID grids remain in memory; only the physical diagnostic plots are
subsampled. The iterative flood fill takes time and storage proportional to the
photon count and adds no Python dependencies.

The script writes `blueprint_domains.txt` next to the blueprint tiles. Its
tab-separated rows report each domain's ID, integer and named termination
status, photon count and percentage, seed coordinates, bounds of the sample
centers, boundary-sample count, and scan edges touched. A boundary sample has a
different-status neighbor in one of the eight directions or touches a scan edge.
The enclosing bounds are not an outline of an irregular domain or a domain
with holes. Any `ACTIVE` domain contains unfinished rays.

The domain figure places sample cells in the original dimensionless `(a,b)`
coordinates, with `b` increasing upward and no interpolation between domain
colors. Click a domain or enter its ID to display its status and zoom to its
extent; the Matplotlib toolbar provides pan, zoom, and Home. Colors repeat after
twenty domains; the selected domain is highlighted in black and its ID is
displayed. An interactive
Matplotlib backend is required for mouse and keyboard controls. Run
`python3 blueprint_analysis.py --no-plots` to write the report without opening
figures. Zooming reveals stored samples; it does not resolve structure between
photon directions or establish a distinct lensed-image order.

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/blueprint_analysis.py` — `diagnose_blueprint`, `label_domains`, `plot_domains`, `plot_heatmaps`; corroboration: `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — termination constants and record fields

This standalone batch raytracer is related to, but distinct from, the
evolution-time raytracing export enabled by
`python -m nrpy.examples.two_blackholes_collide --raytracing-time`. The
black-hole evolution option writes Cartesian metric and Christoffel data on
diagnostic output steps and is currently rejected for CUDA builds. The optional
`--raytracing-static-christoffels` flag changes only the GammaUDD payload on the
last scheduled output; the default remains nonstatic. See
[Black Hole Evolution](black-hole-evolution.md) and
[Geodesics And Raytracing Runtime](../infrastructures/bhah/geodesics-and-raytracing-runtime.md)
for that context.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/diagnostics/diagnostics.py` — `enable_static_christoffels` scheduling; corroboration: `nrpy/infrastructures/BHaH/diagnostics/output_raytracing_data.py` — static GammaUDD selection

For a numerical-spacetime dataset, first generate the evolution project and
its slice-combining workflow:

```bash
python -m nrpy.examples.two_blackholes_collide \
  --raytracing-time 30.0 0.4 \
  --raytracing-data-mode g4DD \
  --raytracing-coord-system SinhCylindricalv2n2
cd project/two_blackholes_collide
./run_raytracing_data_pipeline.sh
```

The generated pipeline copies `combine_raytracing_time_slices.py`, writes
`raytracing_run_metadata.json`, runs the evolution, and combines the full
logical-grid slice payload into `project/raytracing_data/*.bin`. The numerical
single and batch generators then accept that combined filename, the
`SinhCylindricalv2n2` domain tuple, the final stored-time bound, and the nominal
`--dt-spacetime-data` used for synthetic temporal-stencil edge times. Actual
slice-to-slice time gaps are approximately that nominal spacing, but the first
stored slice time is read from the `.bin` and printed at runtime; it is not a
CLI or `.par` input. Stored slice-table times remain authoritative and may be
near, rather than exactly equal to, nominal evolution intervals.

Evidence (public/scientific contract): `nrpy/examples/photon_single_geodesic_integrator_numerical.py` and `photon_batch_geodesic_integrator_numerical.py` — CLI help and runtime parameters; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py` — slice-table loading and time-window construction

```bash
python -m nrpy.examples.photon_single_geodesic_integrator_numerical \
  --bin-name DATASET.bin \
  --coord-system-numerical SinhCylindricalv2n2 \
  --domain 30.0 0.075 0.05 1.0 4.0 \
  --t-numerical-end 30.0 \
  --dt-spacetime-data 0.4 --t-start 10.0 \
  --observer-position 0.0 -10.0 0.0 \
  --observer-look-forward 0.0 1.0 0.0 \
  --observer-up 0.0 0.0 1.0 \
  --escape-radius 25.0 \
  --eom geodesic --interpolation-method g4DD
```

The numerical batch generator has the same numerical-data arguments and adds
the batch visualization/blueprint workflow. Both generators currently require
the numerical `.bin` file to be available under `project/raytracing_data/`.
Its optional `--spacetime-project blackhole_spectroscopy` changes the printed
time-slice combination command to read spectroscopy output. The default printed
command still reads `two_blackholes_collide` output. This option does not change
which combined `.bin` file the photon executable reads; `--bin-name` selects
that file. Spectroscopy output for this photon generator must use
`SinhCylindricalv2n2` and matching `--domain` values.
Both accept `--eom geodesic|normalized` and share the upper-only
`--evolution-measure-max` cutoff (default `3`). For accepted-state normalization
sidecars, direct EOM stores `|g_{mu nu} p^mu p^nu|`; normalized EOM stores
`|exp(2u)(gamma^{ij} Pi_i Pi_j - 1)|`, putting the normalized residual on the
direct-momentum scale. Numerical single- and batch-photon integration retains
the spatial center selected at the start of each attempted RKF45 step and the
temporal center selected at its predicted midpoint. Every stage and the
accepted-state metric refresh reuse those centers.

```bash
python -m nrpy.examples.photon_batch_geodesic_integrator_numerical \
  --bin-name DATASET.bin \
  --coord-system-numerical SinhCylindricalv2n2 \
  --domain 30.0 0.075 0.05 1.0 4.0 \
  --t-numerical-end 30.0 \
  --dt-spacetime-data 0.4 --t-start 10.0 \
  --observer-position 0.0 -10.0 0.0 \
  --observer-look-forward 0.0 1.0 0.0 \
  --observer-up 0.0 0.0 1.0 \
  --observer-fov 1.0 1.0 --scan-density 64 \
  --escape-radius 25.0 \
  --eom geodesic --interpolation-method g4DD
```

Evidence (public/scientific contract): `nrpy/examples/photon_single_geodesic_integrator_numerical.py` and `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — CLI and parameter wiring; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` and `numerical_interpolation.py` — generated integration and interpolation behavior

### Fixed Temporal Stencils For Numerical Photons

The numerical single- and batch-photon integrators select one temporal center
for each attempted RKF45 step. Normalized evolution selects the retained time
slice nearest `t_start_of_step + h/2`; direct evolution uses
`t_start_of_step + h*p0_start/2`. All six stages interpolate at their actual
events using that node set. Retries recompute the center after reducing the
step. The accepted-state metric refresh retains the accepted trial's center.

For positive temporal half-width `n`, the time-window manager derives a runtime
coordinate-time limit just below `(2*n-1)*delta_safe`, rounded toward zero with
`nextafter`. Here `delta_safe` is the minimum of the nominal retained-node
spacing, actual retained-slice gaps, and synthetic endpoint joins. The manager
recalculates this limit for each executable run. The configured `rkf45_h_max` retains its integration-
parameter units and remains a separate limit. For normalized EOM, the temporal
cap is `abs(h) <= H_t`; for direct EOM it is
`abs(h) <= 0.9*H_t/abs(p0_start)`. Initialization and the finalizer share
the cap calculation. A cap below `rkf45_h_min` stops that photon with
`FAILURE_GENERIC` rather than increasing its step beyond the permitted value.

The direct conversion predicts coordinate-time travel; changing `p^0`
within a step can invalidate that estimate. No additional per-stage temporal
containment check is performed. Existing interpolation failures and retry
limits retain their meaning. Fast-light geometry always selects its center
at `t_start` and skips the new travel-based temporal cap. Temporal half-width
zero keeps one-node sampling without a finite-stencil step cap. Frozen
endpoint dispatch and synthetic endpoint-node rules remain in effect.

Single-photon trial and stage files append `temporal_center_slice`,
`temporal_center_time`, `temporal_stencil_first_time`,
`temporal_stencil_last_time`, `temporal_max_delta_t`,
`h_trial_before_temporal_cap`, and `h_trial_after_temporal_cap`.
The last two columns describe the current attempt, including the first one;
`h_trial_after_temporal_cap` equals `h_trial`. Stage stencil endpoints identify
the nodes actually sampled, reducing to one stored time for frozen-endpoint
sampling. Trial stencil endpoints describe the selected center's full node set.
The existing `h_error_controller` trial column precedes mapped-window and
temporal caps; `h_proposed` includes both. Batch output gains no corresponding
diagnostic columns. Numerical Brent refinement holds one temporal center for
the full saved crossing interval without shortening Brent's requested trial
duration.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py` — `time_window_manager_numerical_temporal_step_limit` and `time_window_manager_numerical_stencil_for_center`; corroboration: `numerical_interpolation.py` — `register_CFunction_numerical_interpolation`; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — `rkf45_cap_numerical_step`; `single_integrator_numerical.py`, `batch_integrator_numerical.py`, and `find_plane_crossings_brent.py` — retained centers and diagnostic output

### Metric-Gradient Tolerances For Numerical Photons

With `--interpolation-method g4DD` or `g4DD_d0`, both numerical photon generators
accept optional `--l-sense LENGTH` and `--tol-scale-min SCALE`. Both generated
runtime parameters, `L_sense` and `tol_scale_min`, default to one and can also be
set in the generated parameter file. For example, adding
`--l-sense 1 --tol-scale-min 0.01` allows both absolute and relative RKF45
tolerances to tighten by up to a factor of one hundred.

For each attempted step, the integrator takes the maximum absolute value of
the forty first metric derivatives across all six RK stages. Finalization uses
`q = L_sense * maximum` and
`tol_scale = max(tol_scale_min, 1 / (1 + q))`, and multiplies both base tolerances
by `tol_scale`. The maximum is reset at the first stage of every new attempt.
It measures variation in the chosen Cartesian coordinates, not physical
curvature. `L_sense` is a length in the coordinate units; `tol_scale_min` is a
dimensionless positive value no greater than one.

The default `tol_scale_min = 1` preserves the original RKF45 tolerances and
skips the derivative scan and the batch integrator's additional array allocation.
The single-photon integrator uses a local scalar. `L_sense = 0` also disables
scaling. Both parameters must be finite. The CLI options require a metric
interpolation method; `GammaUDD` retains its existing tolerances. Analytical
programs, the separate normalized log-energy tolerance, and
Brent plane refinement retain their existing calculations. See
[Geodesics And Raytracing Runtime](../infrastructures/bhah/geodesics-and-raytracing-runtime.md)
for the division of calculations between numerical integration and finalization.

Single-photon runs with trial debugging enabled append the following columns:

| File | Added columns |
| --- | --- |
| `rkf45_stages.txt` | `stage_gradient_max`, `stage_gradient_component`, `trial_gradient_max_so_far` |
| `rkf45_trials.txt` | `trial_gradient_max`, `gradient_max_stage`, `gradient_max_component`, `q`, `tol_scale`, `atol_effective`, `rtol_effective`, `interpolation_status` |

Derivative names include both metric indices and direction, such as `g00_t`.
The controlling stage is an RK execution-stage number, not a chronological
ordering of stage times. The trial file's effective tolerances come directly
from finalization. Its comment after the column header records `L_sense`,
`tol_scale_min`, and the base tolerances. Disabled scaling records a scale of
one, the base tolerances, and NaN for uncomputed gradient values. Interpolation
failures retain their trial failure information and record NaN for the unused
gradient and tolerance calculations; uncomputed stages and components are
`-1` and `NOT_COMPUTED`. Only successfully interpolated stages have stage rows.
`interpolation_status` records the status before finalization, so a retryable
interpolation failure remains distinguishable from an RKF45 error rejection.
`--disable-rkf45-trial-debug` omits the diagnostic code while leaving tolerance
scaling available.

Evidence (descriptive behavior): `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` and `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — CLI defaults and metric-method registration; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` and `single_integrator_numerical.py` — per-trial maxima, parameter validation, and single-photon diagnostic columns; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — runtime parameters, tolerance scaling, and diagnostic values

### Consecutive Minimum RKF45 Attempts

All four photon generators accept the optional
`--rkf45-max-consecutive-min-steps N` argument. The default is `100`;
`0` disables this limit. The generated runtime parameter is
`rkf45_max_consecutive_min_steps` in the existing `.par` file. Append
`--rkf45-max-consecutive-min-steps 500` to permit a longer sequence, or
`--rkf45-max-consecutive-min-steps 0` to disable it. The same setting
applies to numerical dynamical and fast-light runs.

Each complete RKF45 attempt at `abs(h_used) <= rkf45_h_min` increments a
separate per-photon count, including accepted attempts and retryable
interpolation failures. An attempt above that magnitude resets the count;
acceptance alone does not. The comparison uses the attempted step before
the controller proposes its next value, and it handles negative steps in
normalized backward evolution. At a positive limit, the photon stops with
`FAILURE_RKF45_MIN_STEP_LIMIT` (value `11`) if it would otherwise continue.
Accepted threshold attempts retain their state and normal physical-event
checks; existing terminal errors and physical stops take precedence. A
rejected threshold attempt retains the previous accepted state.

The limit is independent of `--rkf45-max-retries`, whose counter resets
after acceptance. It bounds minimum-size attempts without establishing
black-hole capture or stopping steps that remain above the floor. Blueprint
analysis labels the numerical failure, and the renderer shows it through the
existing opt-in `--debug` overlay. It receives no puncture-association color.

The main blueprint record layout and version remain unchanged. Updated
readers accept existing files of that version, while older readers reject the new status.
Regenerate and rebuild existing photon projects to obtain the new counter
and generated visualization files.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — attempted-step counting and parameter default; `batch_integrator_analytical.py`, `batch_integrator_numerical.py`, `single_integrator_analytical.py`, and `single_integrator_numerical.py` — stopping after physical checks; corroboration: the four photon example generators — `--rkf45-max-consecutive-min-steps`; `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py`, `blueprint_analysis.py`, and `render_lensed_image.py` — final-status recognition and optional debug color

### Fixed-Time Geometry Comparison

The numerical single-photon and batch examples accept optional `--fast-light`,
disabled by default. Add this switch to either numerical command to evaluate
geometry at `t_start` throughout photon propagation. Both `--eom geodesic`
and `--eom normalized` support it with `g4DD`, `g4DD_d0`, and `GammaUDD`.
For interior sampling times, the RHS retains the numerical spacetime's
dynamical temporal derivatives or stored Christoffels. Existing endpoint
rules still apply. Photon integration parameters and coordinate-time trackers
continue advancing, so crossing intervals and propagation limits remain
available. See [Fixed-Time Geometry Sampling](../infrastructures/bhah/geodesics-and-raytracing-runtime.md#fixed-time-geometry-sampling)
for the separation of scheduling slots from numerical-data mapping.

This is a generator option, not a generated executable or parameter-file
switch. Generate separate comparison projects using, for example,
`--outdir project/time_dependent` without the flag and
`--outdir project/fixed_time --fast-light` with it. The project basename is
the same in both cases. Keep the dataset, `t_start`, camera, sampling density,
and numerical tolerances the same when comparing the resulting images or
single-photon trajectories. Single-photon `trajectory.txt` and plane-crossing
times continue recording the evolving photon time, even though the geometry
is evaluated at `t_start`. Normalization, enabled trial/stage diagnostics,
axial angular momentum diagnostics, and plane refinement use the same selected
sampling mode.

Evidence (descriptive behavior): `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — `--fast-light`, `project_dir`, numerical registration arguments; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — `register_CFunction_numerical_interpolation`, fixed sampling time

Evidence (descriptive behavior): `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — `--fast-light`, interpolation/refinement/integrator registration arguments; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — `single_integrator_numerical`, geometry-slot selection and final normalization

### Puncture Association And Apparent-Horizon Containment

The numerical batch and single-photon generators accept optional
`--puncture-diagnostics-dir PATH`. Supply `BHaHAHA_diagnostics.ah1.gp`, `.ah2.gp`, `.ah3.gp`, and their
`h.tNNNNNNN.ahH.gp` Cartesian apparent-horizon surfaces from the same spacetime
evolution. The generator saves the absolute directory in the generated `.par`
parameter `puncture_diagnostics_dir`; its empty default disables both association
and containment. No additional containment option is needed.

The batch executable classifies only final evolution-measure terminations.
Before the first common-horizon detection, the executable orders the individual horizons by
Cartesian centroid distance. It checks the nearer surface first, then the
second if the first does not establish containment. A containing surface
supplies the puncture ID. If both checks demonstrate outside, retain the
nearest-centroid ID; if either check is unavailable and neither establishes
inside, retain that ID and record `HORIZON_NOT_CHECKED`. An exact distance tie checks
horizon one first and remains unclassified unless containment identifies a
hole. At or after the first common-horizon diagnostic time, assign
`merged_puncture` and test only the common surface. The detection time comes
from the first row of `.ah3.gp`, never a fixed merger time.

Choose each horizon's nearest saved coordinate time, with the earlier row
winning a time tie; do not interpolate in time. Dynamical runs use final photon
time, while fast light uses `t_start`. Negative lookup times use zero. A time
outside a horizon's recorded interval gives `HORIZON_NOT_CHECKED`. Missing or malformed
selected surface files produce an error. Surface filenames use the diagnostic
iteration, not coordinate time. The helper reconstructs the solver's radial
origin from Cartesian rings, since the reported centroid can differ, then
compares photon radius with quadratic Lagrange interpolation of the radial grid,
using three samples per angular direction.
Azimuth is periodic and polar neighbors reflect across the pole; exact polar
directions use the mean nearest ring. Equality within saved-coordinate roundoff
counts as inside. This approximates final-point containment, not a first
horizon crossing along the trajectory, and does not alter photon evolution.

Each `light_blueprint_puncture_XX_YY.bin` stores ten bytes per main blueprint
record, without a header or padding, in the same order:

| Byte | Field | Values |
| --- | --- | --- |
| First | `puncture_id` | `0` unclassified, `1` puncture_1, `2` puncture_2, `3` merged_puncture |
| Second | `horizon_status` | `0` outside, `1` inside, `2` unavailable (`HORIZON_NOT_CHECKED`) |
| Third through tenth | `radial_gap` | Native eight-byte double: photon radius minus the selected horizon radius; negative inside, positive outside, NaN when undefined or unavailable |

Other termination conditions and nonfinite endpoints receive `(0, 2, NaN)`.
The main blueprint format is unchanged. `PUNCTURE_RECORD_DTYPE` in
`blueprint_config_and_schema.py` describes the combined records for NumPy
readers. `blueprint_io.read_puncture_records()` maps all three fields;
`read_puncture_labels()` continues to provide IDs for image coloring. Older
one-byte and two-byte puncture files require photon reruns and matching updated
readers. The gap is a coordinate radial comparison, not proper distance.
The terminal prints a compact per-tile summary: photon counts and
percentages of evolution-measure stops by ID, inside/checked counts and
percentages for each ID, unavailable counts, and totals. Inside percentages
exclude not-checked records.

For the numerical single-photon generator, the same directory option enables
one check of the last accepted state in `trajectory.txt`, for any final stop
reason. It performs no horizon checks during the RKF45 loop. The terminal prints
the trajectory stop reason, photon and lookup times, association, containment
status, selected horizon time, and signed coordinate radial gap
`radial_gap = r - r_AH(theta, phi)`. A positive gap is outside and a negative gap
is inside; the containment flag includes the existing surface-roundoff allowance.
This is a radial coordinate comparison about the recovered solver origin, not a
shortest or proper distance.

The executable writes one data row to `horizon_diagnostics.txt`, with columns:

```text
trajectory_stop_status photon_time lookup_time puncture_id horizon_status horizon_time radial_gap
```

The IDs and status values match the batch records. Dynamical and fast-light
lookup times follow the rules above. No accepted state or unavailable coverage
gives `HORIZON_NOT_CHECKED`; the shared classifier also uses this value for
invalid endpoints. A missing selected surface still produces an error. An unclassified distance tie has no selected horizon
time or gap; the radial gap is also undefined at the reconstructed origin.
Undefined values appear as `nan`. The diagnostic describes the last accepted
trajectory state before Brent refinement selects a terminal-plane crossing;
`plane_crossings.txt` retains the separately refined state. Omitting the
directory option disables this output, and `trajectory.txt` keeps its existing
columns.

From a generated photon project containing the main and puncture blueprints,
render these associations with:

```bash
python3 visualize_lensed_image.py --color-by-puncture --pixel-width 2400 --output shadow_by_puncture.png
```

The renderer needs no horizon diagnostic directory; it uses the saved
`puncture_id` field. It requires a matching combined file for every main tile
and validates its record count and both byte fields before starting rendering
workers. Containment status is stored but does not change the color. Label
slices use the main record offsets, including cropped scans and multiple
tiles. Keep label files paired with the main blueprints from the same run;
their raw byte format cannot distinguish a different run's same-sized file.

Only evolution-measure stops use the association colors:

| Label | Association | Color |
| --- | --- | --- |
| `0` | `unclassified` | Gray, `#A0A0A0` |
| `1` | `puncture_1` | Red, `#FF4040` |
| `2` | `puncture_2` | Blue, `#40A0FF` |
| `3` | `merged_puncture` | Gold, `#FFD040` |

Escaped and terminal-plane photons retain their texture colors. Multiple
photons falling in the same output pixel contribute to the existing color
average, so a pixel can contain a mixture of association colors. Add the
existing `--debug key` option to include the color key and failure diagnostics.
Puncture colors replace the generic evolution-measure debug color; failure
debug colors remain active. With `--color-by-puncture` omitted, rendering
neither requires nor reads the label files.

Newly generated photon projects receive the updated visualization modules.
For an existing project, copy `blueprint_config_and_schema.py`,
`blueprint_io.py`, `render_lensed_image.py`, and `visualize_lensed_image.py`
from `nrpy/examples/geodesic_visualizations/` together into that project.
Existing runs without combined records, including legacy one-byte puncture
files, require another photon integration with `--puncture-diagnostics-dir`.
The reader rejects the old size with a regeneration message. The main blueprint
alone does not retain the Cartesian endpoints needed for these checks.

Evidence (descriptive behavior): `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — `--puncture-diagnostics-dir`, parameter default; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/puncture_diagnostics.py` — `puncture_diagnostics`; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` — final association output; `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — optional diagnostic directory; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — final accepted-state text output

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — `--color-by-puncture`; corroboration: `nrpy/examples/geodesic_visualizations/blueprint_io.py` — `read_puncture_labels`; `nrpy/examples/geodesic_visualizations/render_lensed_image.py` — `generate_static_lensed_image`, `PUNCTURE_COLOR_INFO`; `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — puncture byte values and filename template

## Sources

- [massive_single_geodesic_integrator_analytical.py](../../nrpy/examples/massive_single_geodesic_integrator_analytical.py) - `project_name`, `main_single`, `single_integrator_analytical`, `gsl-config`; official GSL [Using the Library](https://www.gnu.org/software/gsl/doc/html/usage.html) - `Compiling and Linking`
- [photon_single_geodesic_integrator_analytical.py](../../nrpy/examples/photon_single_geodesic_integrator_analytical.py) - `project_name`, `main_single`, observer-tetrad initialization, `rkf45_stage_update`, `trajectory.txt`
- [single_integrator_analytical.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py) - accepted-step event checks and `plane_crossings.txt` output
- [photon_batch_geodesic_integrator_analytical.py](../../nrpy/examples/photon_batch_geodesic_integrator_analytical.py) - `--outdir`, `--cuda`, `parallelization_mode`, `vis_command`, `blueprint_command`; official NVIDIA [NVCC guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html) - `NVCC: The NVIDIA CUDA Compiler`
- [photon_single_geodesic_integrator_numerical.py](../../nrpy/examples/photon_single_geodesic_integrator_numerical.py) - numerical dataset CLI, terminal/nonterminal plane options, `register_CFunction_numerical_interpolation`, `--t-start`
- [single_integrator_numerical.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py) - single-photon accepted-step event checks and `plane_crossings.txt` output
- [event_detection_manager_kernel.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py) - plane crossing reconstruction and termination decisions
- [photon_batch_geodesic_integrator_numerical.py](../../nrpy/examples/photon_batch_geodesic_integrator_numerical.py) - numerical dataset CLI, `combine_raytracing_time_slices.py`, batch visualization workflow
- [batch_integrator_numerical.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py) - `batch_integrator_numerical`, fixed geometry window with advancing photon scheduling times
- [puncture_diagnostics.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/puncture_diagnostics.py) - `puncture_diagnostics`, nearest-time horizon association and final-point containment
- [numerical_interpolation.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py) - `register_CFunction_numerical_interpolation`, `fast_light`
- [time_window_manager_numerical.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py) - `time_window_manager_numerical_temporal_step_limit`, `time_window_manager_numerical_temporal_node_time`, `time_window_manager_numerical_stencil_for_center`
- [find_plane_crossings_brent.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/find_plane_crossings_brent.py) - `find_plane_crossings_brent`, numerical-data mapping during fixed-time refinement
- [normal_observer_log_energy.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/normal_observer_log_energy.py) - direct-EOM `ln|alpha p^0|` helper
- [rkf45_finalize_and_control_kernel.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py) - consecutive minimum-size attempt counting and runtime limit
- [combine_raytracing_time_slices.py](../../nrpy/infrastructures/BHaH/diagnostics/combine_raytracing_time_slices.py) - `InputSliceInfo`, `parse_args`, `--run-metadata`
- [visualize_trajectory.py](../../nrpy/examples/geodesic_visualizations/visualize_trajectory.py) - `visualize_trajectory`, `plot_trajectory`
- [blueprint_config_and_schema.py](../../nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py) - `BLUEPRINT_DTYPE`, `STOP_CONDITION_TERMINAL_PLANE`
- [main_batch.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py) - `main`, photon row counts, global sample coordinates, and saved scan bounds
- [set_initial_conditions_kernel.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/set_initial_conditions_kernel.py) - `set_initial_conditions_kernel`, camera sample mapping and caller-selected row counts
- [blueprint_io.py](../../nrpy/examples/geodesic_visualizations/blueprint_io.py) - `read_blueprint_header`, `iter_blueprint_chunks`
- [render_lensed_image.py](../../nrpy/examples/geodesic_visualizations/render_lensed_image.py) - `generate_static_lensed_image`, `_process_blueprint_tile`, `_load_texture`
- [visualize_lensed_image.py](../../nrpy/examples/geodesic_visualizations/visualize_lensed_image.py) - `main`, `light_blueprint_{i:02d}_{j:02d}.bin`, `urlretrieve`
- [blueprint_analysis.py](../../nrpy/examples/geodesic_visualizations/blueprint_analysis.py) - `diagnose_blueprint`, `plot_heatmaps`
- [two_blackholes_collide.py](../../nrpy/examples/two_blackholes_collide.py) - `--raytracing-time`, `--raytracing-data-mode`, `--raytracing-static-christoffels`

## See Also

- Parent: [Examples](index.md)
- See also: [Black Hole Evolution](black-hole-evolution.md)
- Depends on: [Geodesics](../equations/general-relativity/geodesics.md)
- Depends on: [Geodesics And Raytracing Runtime](../infrastructures/bhah/geodesics-and-raytracing-runtime.md)

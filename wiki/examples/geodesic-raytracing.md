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

All generation, build, executable, trajectory, and rendering commands on this
page are manual/source-supported. Neither GitHub workflow nor the local full-CI
helper invokes these five generators, so these routes remain manual/source-supported.

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

Claim evidence:
- Claim: Both single-photon generators refine accepted crossing steps with fixed-anchor RKF45 Brent trials and write the refined states to `plane_crossings.txt`.
- Role: descriptive behavior
- Deciding authority: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py` and `single_integrator_numerical.py` — `plane_crossings.txt` output and EOM state columns
- Corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py` — accepted-step crossing capture; `find_plane_crossings_brent.py` — analytical and numerical refinement; both single-photon generators — Brent-tolerance CLI

Both single-photon generators enable RKF45 trial and stage diagnostics by
default. Their generated executables write adaptive-step controller values to
`rkf45_trials.txt` and all six metric, connection, and right-hand-side stage
evaluations for each trial to `rkf45_stages.txt`; the numerical version records
the interpolated geometry used at each stage. Pass
`--disable-rkf45-trial-debug` during project generation to omit this diagnostic
code; use `--enable-rkf45-trial-debug` to select it explicitly.

Claim evidence:
- Claim: The analytical and numerical single-photon generators construct unit camera-tetrad-energy initial momentum, run the split RKF45 calculation, apply the upper-only log-energy cutoff, write `trajectory.txt`, enable trial/stage diagnostics by default, and support optional terminal and nonterminal plane crossings recorded in `plane_crossings.txt`.
- Role: descriptive behavior
- Deciding authority: `nrpy/examples/photon_single_geodesic_integrator_analytical.py` and `photon_single_geodesic_integrator_numerical.py` — plane options and debug defaults; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py` and `single_integrator_numerical.py` — generated C behavior
- Corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py` — crossing and termination checks

`trajectory.txt` is the input file for the single-ray visualization. The
massive file header is `# proper_time t x y z u^t u^x u^y u^z`. Direct photon
output uses `# lambda t x y z p^t p^x p^y p^z L_normal`; normalized output uses
`# lambda t x y z u Pi_1 Pi_2 Pi_3 L_normal`. Numerical photon output appends a
`norm` column containing the accepted-state constraint residual.
`visualize_trajectory.py` loads the text file with NumPy, validates it is present
and non-empty, prints initial and final spatial positions plus the accumulated
affine or proper parameter, then saves `trajectory_plot.png` and displays a
Matplotlib 3D path with a schematic `r=2M` reference sphere.

Claim evidence:
- Claim: `trajectory.txt` is the single-ray visualization handoff with the documented state headers, and `visualize_trajectory.py` validates the input before plotting it.
- Role: generated evidence
- Deciding authority: `nrpy/examples/geodesic_visualizations/visualize_trajectory.py` — `visualize_trajectory`; `nrpy/examples/photon_single_geodesic_integrator_analytical.py` — trajectory header
- Corroboration: `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — numerical trajectory header

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

Claim evidence:
- Claim: The analytical batch generator defaults to OpenMP, selects CUDA with `--cuda`, derives its project directory from `--outdir`, and exposes tile counts and scan density as angular sampling controls while final PNG pixel width remains a visualization setting.
- Role: public/scientific contract
- Deciding authority: `nrpy/examples/photon_batch_geodesic_integrator_analytical.py` — argument parser and project generation
- Corroboration: `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — header-driven tile and aspect-ratio handling

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

Claim evidence:
- Claim: Image-plane range options select camera directions without reducing the requested column count; cropped scans require one tile, cropped row counts use floor rounding, and single-photon initialization uses one sample at the selected center.
- Role: generated evidence
- Deciding authority: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py` — `main`; `set_initial_conditions_kernel.py` — `set_initial_conditions_kernel`
- Corroboration: `nrpy/examples/photon_batch_geodesic_integrator_analytical.py`, `photon_batch_geodesic_integrator_numerical.py`, `photon_single_geodesic_integrator_analytical.py`, and `photon_single_geodesic_integrator_numerical.py` — range options and parameter defaults; `single_integrator_analytical.py` and `single_integrator_numerical.py` — one-row initialization

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

Claim evidence:
- Claim: Photon batch blueprint files use native same-build binary-layout version 9 with exact image-plane bounds in a 92-byte header and 124-byte photon records; Python `BLUEPRINT_DTYPE` must match the generated C `blueprint_data_t` layout and termination enums.
- Role: generated evidence
- Deciding authority: `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — `BLUEPRINT_SCHEMA_VERSION`, `BLUEPRINT_DTYPE`
- Corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/calculate_and_fill_blueprint_data_universal.py` — generated record layout

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

Claim evidence:
- Claim: `visualize_lensed_image.py` consumes per-tile binary files, discovers tile geometry from headers, maps normalized image fractions to pixels, and delegates final rendering to `render_lensed_image.py`.
- Role: generated evidence
- Deciding authority: `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — `main`; `render_lensed_image.py` — `generate_static_lensed_image`
- Corroboration: `nrpy/examples/geodesic_visualizations/blueprint_io.py` — binary header and record readers

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

Claim evidence:
- Claim: `blueprint_analysis.py` reports termination diagnostics, labels equal-status eight-neighbor domains using every photon across all tiles, writes their summaries to `blueprint_domains.txt`, provides an interactive domain figure and plane/celestial heatmaps, and consumes complete matching normalization sidecars when available.
- Role: descriptive behavior
- Deciding authority: `nrpy/examples/geodesic_visualizations/blueprint_analysis.py` — `diagnose_blueprint`, `label_domains`, `plot_domains`, `plot_heatmaps`
- Corroboration: `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — termination constants and record fields

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

Claim evidence:
- Claim: Evolution-time raytracing export is distinct from the standalone batch raytracer; `--raytracing-static-christoffels` is optional, default-off, and changes only the GammaUDD payload on the last scheduled output.
- Role: public/scientific contract
- Deciding authority: `nrpy/infrastructures/BHaH/diagnostics/diagnostics.py` — `enable_static_christoffels` scheduling
- Corroboration: `nrpy/infrastructures/BHaH/diagnostics/output_raytracing_data.py` — static GammaUDD selection

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

Claim evidence:
- Claim: The numerical pipeline combines full logical-grid slice payloads into a `.bin` container; nominal `--dt-spacetime-data` describes approximate gaps and synthetic edge times, while stored slice-table times remain authoritative at runtime.
- Role: public/scientific contract
- Deciding authority: `nrpy/examples/photon_single_geodesic_integrator_numerical.py` and `photon_batch_geodesic_integrator_numerical.py` — CLI help and runtime parameters
- Corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py` — slice-table loading and time-window construction

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
direct-momentum scale. The numerical batch path locks the spatial interpolation
center selected for each RK trial and reuses that same center for every RK
substage and the accepted-state metric refresh.

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

Claim evidence:
- Claim: Numerical photon generators expose direct and normalized EOM modes with a common upper-only log-energy cutoff and direct-scale-equivalent normalized norm sidecars; numerical batch RK stages reuse one spatial interpolation center per trial.
- Role: public/scientific contract
- Deciding authority: `nrpy/examples/photon_single_geodesic_integrator_numerical.py` and `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — CLI and parameter wiring
- Corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` and `numerical_interpolation.py` — generated integration and interpolation behavior

### Fixed-Time Geometry Comparison

The numerical batch example accepts optional `--fast-light`, disabled by
default. Add this switch to the numerical batch command to generate geometry
evaluations at `t_start` throughout photon propagation. Both `--eom geodesic`
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
and numerical tolerances the same when comparing the resulting images.

Evidence (descriptive behavior): `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — `--fast-light`, `project_dir`, numerical registration arguments; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — `register_CFunction_numerical_interpolation`, fixed sampling time

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
- [numerical_interpolation.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py) - `register_CFunction_numerical_interpolation`, `fast_light`
- [find_plane_crossings_brent.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/find_plane_crossings_brent.py) - `find_plane_crossings_brent`, numerical-data mapping during fixed-time refinement
- [normal_observer_log_energy.py](../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/normal_observer_log_energy.py) - direct-EOM `ln|alpha p^0|` helper
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

- [Examples](index.md)
- [Black Hole Evolution](black-hole-evolution.md)
- [Geodesics](../equations/general-relativity/geodesics.md)
- [Geodesics And Raytracing Runtime](../infrastructures/bhah/geodesics-and-raytracing-runtime.md)

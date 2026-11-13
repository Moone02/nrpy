# Geodesics And Raytracing Runtime

> BHaH runtime pieces for standalone geodesics and evolution-time raytracing export. · Status: confirmed
> Up: [BHaH](index.md)

## Summary

BHaH has standalone single- and batch-photon geodesic programs plus an
evolution-time raytracing-data export. Both analytical and numerical
single-photon programs accept optional terminal and nonterminal planes; the
analytical Brent evaluation still requires the shared crossing functions. The
batch program initializes observer and independent plane parameters, tiles the
angular pixel grid, integrates photon groups with RKF45, and writes per-tile
light-blueprint binary files. The evolution-time export writes mode-selected
Cartesian `g4DD`, `g4DD_d0`, and `Gamma4UDD` time-slice data from a live BSSN
evolution; a combiner validates and stacks those slices for later
numerical-spacetime interpolation.

## Detail

The standalone batch-photon entrypoint is `main` in the photon geodesics package. It
registers angular tile-grid parameters, initializes `commondata`, parses
command-line and parfile input, loops over `(tx, ty)` tile indices, and calls
the selected batch integrator. The shared initializer invoked by that integrator
constructs the observer tetrad, including fallback logic for near-degenerate up
vectors. Each record receives normalized image-sample
coordinates before serialization; image placement does not depend on integer
pixel identity fields.
This path is a standalone geodesic program; it is not the same as
diagnostics emitted by an evolving BHaH spacetime. Single-photon generators use
the particle-independent forwarding `main` from `geodesics/main_single.py`.
Both single-photon generators register optional event-plane parameters and the
crossing functions used by their integrators.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py` — `main`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/calculate_and_fill_blueprint_data_universal.py` — normalized record fields

The shared `set_initial_conditions_kernel` receives one metric evaluated at the
observer event and constructs one validated metric-orthonormal tetrad per call.
It uses that tetrad for every requested ray, writes complete contravariant
`p^mu`, and only then performs any requested normalized-variable conversion.
Batch generators and both single-photon generators initialize each enabled
plane's starting side when registering their shared initializer.
Event-plane bases used by crossing handlers are not the metric tetrad used to
initialize momentum.

`main_batch.main` calculates the photon row count once and passes it through
the selected batch integrator to `set_initial_conditions_kernel`. The
initializer checks the supplied count against the allocated rays. Both
single-photon integrators supply one row and retain one photon, so rectangle
aspect ratio cannot change their allocation or displace their sample from the
selected center.

The parameters `image_plane_crop_enabled`, `a_min`, `a_max`, `b_min`, and
`b_max` describe optional image-plane scanning. The crop flag is false by
default and both ranges default to `[0,1]`. Cropped scans require one tile.
The initializer maps sample centers into the selected rectangle before
constructing momentum. The batch writer applies the same mapping when saving
global image coordinates. Range selection leaves the full-camera fields of
view, metric, tetrad, and event-plane geometry unchanged. Cropped batches use
floor-rounded row counts from the range and tangent half-field-of-view ratios;
uncropped batches retain the existing nearest-integer tile-aware calculation.
The [ray-tracing examples](../../examples/geodesic-raytracing.md) give the
sampling and output-image equations.

The generated ray construction sets the initial camera-tetrad photon energy
magnitude to `E_camera = 1` before any normalized-variable conversion. This
affine normalization is distinct from the hypersurface-normal measure
`|alpha p^0|` used by later termination diagnostics.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/set_initial_conditions_kernel.py` — observer initialization; corroboration: `nrpy/examples/photon_batch_geodesic_integrator_numerical.py`, `nrpy/examples/photon_single_geodesic_integrator_analytical.py`, `nrpy/examples/photon_single_geodesic_integrator_numerical.py`, `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py`, and `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — shared observer and plane-side initialization arguments

`batch_integrator_numerical` evolves numerical-spacetime photon batches. It
registers integration limits and RKF45 controls in `commondata`, allocates the
Structure-of-Arrays photon state, step-size, status, saved crossing steps, and
result buffers, and uses the TimeSlotManager to process active rays by
coordinate-time slots. The split kernels exchange component arrays for photon
state, metric, metric derivatives or Christoffels, RKF45 stage derivatives,
integration parameter, retries, and termination state. After an accepted RKF45
step, direct mode refreshes the metric at the accepted state using the same
spatial and temporal stencil centers used by the RK stages. The integrator then
fills a per-ray log-energy array: normalized mode copies `u`, while
direct mode computes `ln|alpha p^0|` with `normal_observer_log_energy`. The event
manager applies the shared upper-only `evolution_measure_max` cutoff. Its
normalization sidecar stores `|C_direct|` for direct EOM and
`|exp(2u)(C_normalized - 1)|` for normalized EOM.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` — `batch_integrator_numerical`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/time_slot_manager_helpers.py` — `TimeSlotManager`

The numerical batch terminal display reports retryable rejections in the latest
processed photon chunk. Its `RKF45`, `Spatial`, and `Temporal` percentages all
use the number of attempted photons in that chunk as their denominator.
`Spatial` covers coordinate inversion, spatial-stencil, and spatial-output
failures; `Temporal` covers time-stencil and temporal-interpolation failures.
The integrator saves each trial's incoming status before finalization replaces
retryable interpolation failures with `REJECTED`. Integer category counts sum
to the total rejection count; independently rounded percentages can differ
from the displayed total by one tenth of a percentage point. Terminal failures
and caps on the next accepted step are excluded. These percentages are separate
from the smoothed RKF45 attempt rate and are not cumulative over the run.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` — saved trial statuses, rejection counting, and terminal output in `batch_integrator_numerical`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — retryable interpolation failures and RKF45 error control

All four photon integrators carry an independent per-photon
`consecutive_min_steps` counter. The shared RKF45 finalizer increments it
once per accepted or rejected attempt whose magnitude satisfies
`abs(h_used) <= rkf45_h_min`, including retryable interpolation failures.
An attempted magnitude above the floor resets it; acceptance alone does not.
The comparison precedes any update of the proposed next step, and zero
`rkf45_max_consecutive_min_steps` disables counting. The default positive
limit is `100`, configurable through the generated parameter file or
`--rkf45-max-consecutive-min-steps` in each photon example generator.

The host integrators apply the limit after normal physical stopping checks
and before requeue or retry continuation. A photon that would otherwise
continue at the threshold receives `FAILURE_RKF45_MIN_STEP_LIMIT = 11`;
other terminal reasons retain precedence. Accepted threshold steps retain
their states and event checks; rejected attempts retain the previous accepted
state. Batch counts follow global photon indices across both buffers and
time slots. The counter is separate from the rejection counter, which still
resets on acceptance. Its value is not added to blueprint records. Existing
post-evolution terminal-plane selection retains its trajectory-stop reason.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — `rkf45_max_consecutive_min_steps` and attempted-step counting; `set_initial_conditions_kernel.py` — photon field and status; corroboration: `batch_integrator_numerical.py`, `batch_integrator_analytical.py`, `single_integrator_numerical.py`, and `single_integrator_analytical.py` — preserved counts and stopping order

The RKF45 kernels are deliberately split. `interpolation_kernel` evaluates
spacetime-specialized `g4DD_metric` and `connections` helpers for each ray and
writes 10 metric and 40 connection components into bundles. `calculate_ode_rhs_kernel`
unpacks coordinates, momenta, metric entries, and Christoffels, evaluates the
nine photon RHS expressions, and writes the selected RK stage into
`d_k_bundle`. `rkf45_stage_update` reads the base state, stage derivatives, and
per-ray step size, applies the RKF45 Butcher coefficients for stages 1-5, skips
stage 6, and writes the temporary state for the next stage. The finalization and
control kernel commits accepted states; the accepted-state metric refresh,
log-energy calculation, event manager, and time-slot helpers then decide which
rays remain active and which step sizes advance.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/interpolation_kernel.py`, `calculate_ode_rhs_kernel.py`, and `rkf45_stage_update.py` — generated kernel interfaces; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — acceptance/control

The numerical batch and single-photon generators enable optional metric-gradient tolerance scaling
for `g4DD` and `g4DD_d0` in either photon equation mode. Immediately after each
stage's interpolation, the batch integrator scans the forty Cartesian first
metric derivatives and retains their maximum absolute value for each photon
over all six trial stages. Stage one overwrites the previous maximum, including
on retries and when a working buffer receives a different photon chunk.
Interpolation failures retain their existing rejection handling.
The single-photon integrator performs the same scan using one local scalar
maximum, initialized for every attempt, without an additional allocated array.

Finalization computes `q = L_sense * trial_gradient_max` and
`tol_scale = max(tol_scale_min, 1 / (1 + q))`, then multiplies both base absolute
and relative tolerances by `tol_scale` before calculating the error norm.
`L_sense` has units of length when Cartesian coordinates and coordinate time
have the same units; `tol_scale_min` is dimensionless. These generated runtime
parameters default to one. `tol_scale_min = 1` or `L_sense = 0` skips the scan,
its per-photon allocation, and tolerance scaling. Finite `L_sense >= 0` and
`0 < tol_scale_min <= 1` are required. The normalized log-energy tolerance is
separate and is not multiplied by this scale.

This maximum measures coordinate-dependent metric variation, not curvature.
No additional spacetime fields are interpolated. A Python generation option
selects the additional finalization argument and calculation; analytical
generators retain the existing finalization calls. `GammaUDD`
retains its existing tolerances. Fixed-anchor Brent trials and their plane
distance and parameter-width tolerances do not use this scaling.

With single-photon trial debugging enabled, `rkf45_stages.txt` appends
`stage_gradient_max`, `stage_gradient_component`, and
`trial_gradient_max_so_far`. Components name the metric pair and Cartesian
derivative direction, such as `g00_t` or `g12_z`. `rkf45_trials.txt` appends
`trial_gradient_max`, `gradient_max_stage`, `gradient_max_component`, `q`,
`tol_scale`, `atol_effective`, `rtol_effective`, and `interpolation_status`. The shared finalization
function writes the actual effective tolerances into its trial diagnostic
structure. The trial file also records the sensing length, minimum scale,
and base tolerances in a comment following its column header.

Disabled scaling records `tol_scale = 1`, the base tolerances, and NaN for
uncomputed gradient quantities, with stage `-1` and component `NOT_COMPUTED`.
On interpolation failures, finalization records NaN for the unused gradient
maximum, `q`, scale, and effective tolerances; stage and component also use the
uncomputed markers. The trial failure and retry information remain available.
`interpolation_status` preserves the status before finalization changes a
retryable geometry failure to `REJECTED`, distinguishing it from an RKF45
error-estimate rejection whose interpolations succeeded.
Stage rows are written only for successful interpolations. These diagnostics
do not add geometry evaluations or modify Brent refinement.

Evidence (descriptive behavior): `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` and `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — metric-method registration and optional `--l-sense` and `--tol-scale-min`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` and `single_integrator_numerical.py` — stage maxima and runtime parameter validation; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — `enable_metric_gradient_tolerances`, parameter registration, local tolerance multiplication, and optional diagnostic fields

`event_detection_manager_kernel` applies log-energy and coordinate-radius
termination rules and checks enabled planes after accepted RKF45 steps. In
analytical CUDA batches, device threads record crossings; the batch integrator
copies those records to the host for Brent refinement and plane projection.
Analytical evolution has no numerical time-window stop. A plane sign change saves the accepted state before the
crossing and both integration parameters around that step. Rejected steps do
not create plane candidates. The integration parameter is affine parameter
for direct EOM and coordinate time for normalized EOM; state component zero
stores the other quantity. Plane handlers compute local coordinates and apply
the terminal plane's radial bounds after numerical refinement.

Numerical batches save the accepted state before each detected crossing, then
continue evolving the photon. After evolution, a fixed-anchor RKF45 step and
Brent refinement locate each saved crossing. The numerical batch selects the
first terminal crossing within the configured radial bounds and retains a
nonterminal crossing only if it precedes that selected terminal crossing.
Normalization and axial angular momentum diagnostics use the refined plane
states. The selected terminal crossing sets the blueprint termination status,
coordinate time, and affine parameter before final blueprint calculation.

Both batch integrators write separate sparse
`light_blueprint_non_terminal_crossings_XX_YY.bin` and
`light_blueprint_terminal_crossings_XX_YY.bin` files. Each native record holds
the tile-local photon index, crossing integration parameter, two local plane
coordinates, nine state components, and Brent iteration counts. The numerical
image blueprint stores axial angular momentum, its measurement time, and signed
plane residual at the refined nonterminal crossing.
Both single-photon integrators write `plane_crossings.txt` after evolution.
The rows contain refined plane states, local coordinates, Brent iteration
counts, and signed plane distances. Analytical batches transfer CUDA crossing
records to the host before refinement.

The numerical single-photon integrator explicitly initializes the nonterminal
record with `NON_TERMINAL_PLANE_EVENT` and every terminal record with
`TERMINAL_PLANE_EVENT`. Brent refinement uses that field to select the plane
centre and normal. A zero-initialized terminal record instead selects the
nonterminal geometry, which can reject an otherwise valid terminal crossing.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py` — `event_detection_manager_kernel`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/find_plane_crossings_brent.py` — analytical and numerical refinement and terminal selection; `single_integrator_analytical.py` and `single_integrator_numerical.py` — single-photon crossing output; `batch_integrator_analytical.py` and `batch_integrator_numerical.py` — sparse crossing files and plane diagnostics; `handle_non_terminal_plane_intersection.py` and `handle_terminal_plane_intersection.py` — local coordinates and radius filtering

Blueprint headers carry tile identity/counts, `alpha_w`, `alpha_h`, and exact
`a_min`, `a_max`, `b_min`, and `b_max` scan bounds in binary-layout version 9.
The header is 92 bytes; each native same-build photon record remains 124 bytes.
Records carry nonterminal
crossing coordinates, affine parameter, and coordinate time; terminal-plane
texture coordinates; final angles, termination affine parameter and coordinate
time; normalized image-sample fractions; and axial angular momentum, its
measurement time, and signed plane residual at a refined nonterminal crossing.
Final records admit spatial- and temporal-interpolation failures and
`FAILURE_RKF45_MIN_STEP_LIMIT` alongside the existing physical stops and
numerical failures; internal `ACTIVE` and `REJECTED` states remain invalid
serialized outcomes. The new enum value does not change the record layout
or version. Updated readers accept earlier files of this version; older
readers reject the new value. Analysis names the minimum-step failure, and
the optional renderer debug overlay gives it a distinct color.
The renderer maps the stored scan bounds to the output edges and preserves the
vertical raster flip. The requested pixel width spans the complete stored
`a` range; height follows the selected projected camera-plane proportions,
including the tangent half-field-of-view factors. Output height is rounded
down, with a minimum of one pixel; output resolution does not add photon
samples. Readers require version 9 headers and do not infer scan edges from
sample-center extrema. Plane diagnostics remain available to
`blueprint_analysis.py`. That analysis also consumes optional matching
normalization sidecars and plots their magnitudes by termination status.

Evidence (descriptive behavior): `nrpy/examples/geodesic_visualizations/blueprint_config_and_schema.py` — binary-layout constants and dtype; corroboration: `nrpy/examples/geodesic_visualizations/visualize_lensed_image.py` — image placement

Metric and connection generation is shared by photon and massive geodesic
runtimes. `g4DD_metric` writes the upper-triangular 10-component covariant
metric into a thread-local array, while `connections` writes the 40 unique
Christoffel components. Both unpack only coordinates used by the generated
SymPy expressions and validate the particle-state width by `PARTICLE`.
Massive geodesics use the GSL path instead of the batched photon RKF45 path:
the spacetime-specialized massive-geodesic GSL wrapper casts the GSL parameter
pointer to `commondata_struct`, evaluates metric and connection locally, calls
`calculate_ode_rhs_massive`, and returns `GSL_SUCCESS`.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/g4DD_metric.py`, `connections.py`, and `massive/ode_gsl_wrapper_massive.py` — generated helper registrations; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/massive/calculate_ode_rhs_massive.py` — massive RHS

Numerical-spacetime interpolation is separate from analytic metric evaluation.
`register_CFunction_numerical_interpolation` emits a CPU wrapper that assumes a
mapped `NumericalTimeWindowManager`; for every ray in a chunk it asks the time
window for the temporal stencil, performs `SinhCylindricalv2n2` azimuthal-
symmetry spatial Lagrange interpolation on every mapped slice, then calls
temporal Lagrange interpolation at the photon's coordinate time by default. The time-window
manager owns the read-only mmap window over a combined numerical-spacetime
container and widens each slot by temporal interpolation halo plus backward
RKF45 lookahead. The exporter and combiner preserve the full logical grid,
including ghost-zone points; `InputSliceInfo` describes source slices in the
combiner. Stored slice-table times are authoritative: the first stored time is
loaded into runtime `t_numerical_initial` and printed at startup, while the
user supplies only `t_numerical_end`. The nominal `dt_numerical_spacetime_data`
describes approximate evolution spacing, sets synthetic stencil-node times,
and bounds the spacing used for the temporal-stencil step limit. Retained nodes
use the stored slice-table times, which may be nonuniform.
Numerical integration callers supply fixed native spatial centers and a fixed
retained temporal center for each attempted RKF45 step. The wrapper reuses the
temporal node indices and spatial centers for all stages and the accepted-state
metric refresh. Normalized callers select the time node nearest the predicted
coordinate-time midpoint; direct callers predict that midpoint using the
accepted `p^0`. Retries select again using the reduced step. Standalone
geometry requests can omit the temporal center and select nodes automatically.
Normalized-EOM calls provide the integration parameter, trial step, and RK
stage so the interpolant is evaluated at the actual stage coordinate time;
direct-EOM calls read that time from the stage state.

The time-window manager computes the internal runtime field
`numerical_spacetime_temporal_max_delta_t` from temporal half-width and actual
retained/synthetic node gaps. The finalizer's shared cap helper also bounds
initial photon steps. This temporal limit is separate from `rkf45_h_max`
and the mapped-slot cap. Direct conversion divides by the magnitude of the
current accepted `p^0` with a factor of 0.9; it predicts rather than guarantees
the stage-time span. A half-width of zero or fixed-time fast-light sampling
disables this additional travel cap. Existing frozen endpoint dispatch and
retry classifications remain. Single-photon trial and stage diagnostics record
the current attempted step before and after the temporal cap; stage node bounds
reduce to the frozen endpoint time when no temporal interpolation is performed.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py` — `time_window_manager_numerical_temporal_step_limit` and `time_window_manager_numerical_stencil_for_center`; corroboration: `numerical_interpolation.py` — `register_CFunction_numerical_interpolation`; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — `rkf45_cap_numerical_step`; `single_integrator_numerical.py` and `batch_integrator_numerical.py` — temporal-center selection and step diagnostics

### Fixed-Time Geometry Sampling

The numerical single-photon and batch generators' optional `--fast-light` switch
selects `fast_light=True` when registering numerical interpolation, integration,
and numerical plane-crossing refinement. It is disabled by default and applies
to both `geodesic` and `normalized` equations with `g4DD`, `g4DD_d0`, or
`GammaUDD` data. Every geometry evaluation uses `commondata->t_start`, including
RKF45 stages, accepted-state metric evaluations, plane refinement, and
normalization or angular-momentum diagnostics. For interior sampling times,
the original dynamical metric time derivatives or stored Christoffels remain
in the RHS. Existing endpoint and synthetic-node rules still apply; the option
does not add a temporal-derivative-zeroing operation.

Photon states and integration parameters still advance. In direct evolution,
coordinate time is state component zero; in normalized evolution, coordinate
time is the independent integration parameter and component zero is affine
parameter. Scheduling slots, crossing intervals, output times, and existing
propagation limits retain those advancing values. The mapping calls use the
slot containing `t_start`, independently of the photons' scheduling slots.
The batch diagnostic slot grids have the same time bounds and spacing as the
main grid, so they reuse its initial slot index. Numerical plane refinement
selects that time's slot in its own grid. Fast-light sampling uses the same
time-slot and numerical time-window helpers as dynamical sampling.

The single-photon integrator keeps its existing checks on the advancing
coordinate-time slot, but fast light reuses `initial_slot_index` for geometry
mapping. Its final normalization also uses that initial slot. Plane refinement
invalidates the recorded mapping slot, so the final diagnostic remaps the
selected sampling-time slot before interpolating geometry. Accepted-state
normalization and enabled trial/stage and axial angular momentum diagnostics
use the same fixed-time interpolation as the RHS. Trajectory and crossing
outputs retain their evolving photon times.

This option fixes the time used to evaluate the dynamical RHS. It does not
construct a stationary metric by discarding its temporal derivatives. The
existing RKF45 error controller, mapped-slot step cap, and photon termination
conditions remain active; fast light disables the additional temporal-stencil
travel cap.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — `register_CFunction_numerical_interpolation`, `fast_light`; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` — `batch_integrator_numerical`, integration and diagnostic mapping calls; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/find_plane_crossings_brent.py` — `find_plane_crossings_brent`; `nrpy/examples/photon_batch_geodesic_integrator_numerical.py` — `--fast-light` registration arguments

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — `single_integrator_numerical`, `fast_light`, geometry-slot selection; corroboration: `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — matching interpolation, refinement, and integration registration

### Puncture Association And Apparent-Horizon Output

`puncture_diagnostics.py` generates local C helpers enabled by a nonempty
`puncture_diagnostics_dir`. The directory contains
`BHaHAHA_diagnostics.ah1.gp`, `.ah2.gp`, `.ah3.gp`, and saved
`h.tNNNNNNN.ahH.gp` surfaces. Table columns one through five give iteration,
coordinate time, and Cartesian centroid. Times must be finite and strictly
increasing, with finite centroids and individual tables beginning at zero.
The common table's first row determines first detection; a comment-only table
indicates no common horizon. Requested invalid tables fail before evolution.

After terminal-plane selection and main blueprint calculation, the integrator
passes final states and saved termination records to
`write_puncture_diagnostics`. Only final evolution-measure stops are classified.
Dynamical lookup uses the final coordinate time, fast light uses `t_start`,
and negative times use zero. Each horizon uses its nearest saved row, with the
earlier row winning a time tie. Outside that table's recorded time interval,
containment receives `HORIZON_NOT_CHECKED`. At or after first common detection, only the common
horizon is tested. Before detection, centroid distances order the two individual
checks; the second is needed only if the first does not establish inside.
A containing surface supplies the ID. If both are outside, retain the nearest
ID; if neither establishes inside and either check is unavailable, retain the
nearest ID with `HORIZON_NOT_CHECKED`. An equal-distance tie uses horizon one first and
retains unclassified unless a surface contains the point.

A needed surface is read once per tile and retained with its diagnostic row.
Its filename uses that row's iteration. The loader infers angular dimensions
from Cartesian rings, validates repeated closing points, and recovers the
solver origin using the known cell-centered angular directions and a
three-variable least-squares solve. It validates a positive radial grid,
releases the Cartesian points, and retains the origin and radial samples.
Containment compares photon radius with quadratic Lagrange interpolation using
three radial samples per angular direction, periodic azimuth, and polar
reflection; exact poles use the nearest ring's mean radius. The uniform-grid
Lagrange helper evaluates weights in angular grid-index coordinates.
A relative allowance for saved-coordinate roundoff includes surface equality.
Missing or malformed selected surfaces fail explicitly. This is an approximate
nearest-time final-point check, not a trajectory's first horizon crossing.

`main_batch.main` supplies `light_blueprint_puncture_XX_YY.bin` per tile.
The helper writes bounded chunks of ten-byte records without a header or
padding: `puncture_id` (one byte: zero unclassified, one or two individual,
three common), `horizon_status` (one byte: zero outside, one inside, two
unavailable (`HORIZON_NOT_CHECKED`)), and `radial_gap` (native eight-byte double).
Non-evolution terminations and invalid endpoints receive `(0, 2, NaN)`.
Keep these files paired
with their main tile's ordering. The main blueprint and photon equations are
unchanged. The helper prints association counts and percentages among
evolution-measure stops, inside/checked percentages excluding unavailable
checks, status totals, and the number of loaded surfaces, then releases its
diagnostic allocations. Visualization validates IDs and statuses, rejects
infinite gaps, and colors by ID; older one-byte and two-byte puncture files
require regeneration with matching updated readers.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/puncture_diagnostics.py` — `puncture_diagnostics`, generated table/surface readers, nearest-time containment, combined writer and summary; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py` — final-state output call; `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py` — numerical diagnostic filename

The numerical single integrator requests shared readers and classification
helpers with `puncture_diagnostics(include_batch_writer=False)`. It loads the
diagnostic tables once, calls the classifier once on its last accepted state
after integration, and releases cached surfaces during cleanup. The check uses
the final photon coordinate time for direct or normalized dynamical equations,
and fixed `t_start` for fast light, with negative lookup times clamped to zero.
All final stop reasons are eligible when an accepted state exists. There is no
per-step horizon calculation.

The internal C result also carries the selected surface time and
`radial_gap = r - r_AH(theta, phi)`. Batch serialization writes the gap after
the ID and status bytes; the selected surface time remains single-photon text
output only. A positive gap means outside and a negative gap
means inside in radial coordinates about the recovered solver origin; this is
not proper distance. An undefined direction at that origin gives a NaN gap.
An unclassified distance tie or unavailable classification has no selected
surface time or gap. Containment retains the existing roundoff allowance.

The single integrator prints a compact final accepted-state report and writes
one row to `horizon_diagnostics.txt`: trajectory stop status, photon time,
clamped lookup time, puncture ID, horizon status, selected surface time, and
radial gap. No accepted state gives `HORIZON_NOT_CHECKED`. This output precedes
terminal-plane refinement so its position and photon time match the final
`trajectory.txt` row. The trajectory format and integration equations remain
unchanged; an empty diagnostic directory disables the new output.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — final accepted-state classifier call, text output, and cleanup; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/puncture_diagnostics.py` — shared result and optional batch writer; `nrpy/examples/photon_single_geodesic_integrator_numerical.py` — directory CLI and parameter default

### Numerical Interpolation Diagnostics And Boundaries

The numerical batch integrator can enable `perform_synthetic_slice_check` to
record synthetic temporal-stencil nodes whose times lie strictly below `0` or
above `t_numerical_end`. Direct first/final-slice endpoint dispatch does not
count, and normalization or `L_z` diagnostic interpolations do not count. The
wrapper records the first successful RKF45 interpolation request time for each
photon and boundary. It writes one separate 26-byte tile sidecar record per
photon: a `uint64_t` photon index, two `uint8_t` lower/upper-use flags, then
lower and upper `f64` request times. A false flag has a `NaN` time. Records
have no file header and use native byte order.
The numerical single-photon integrator uses the same wrapper check for RKF45
stage interpolations and prints its lower/upper flags and first request times
once the run finishes; it does not write a sidecar file.

The spatial helper performs tensor-product Lagrange interpolation only in the
two non-azimuthal native coordinates. It obtains spatial metric derivatives
from analytic derivatives of the same uniform-grid basis, rotates tensors from
the stored reference-azimuth plane to the target azimuth, and transforms native
derivatives to Cartesian coordinates with the reference-metric Jacobian. The
temporal helper builds a barycentric Lagrange basis from the actual, potentially
nonuniform slice times. For `g4DD`, analytic derivatives of that temporal basis
provide metric time derivatives; `g4DD_d0` and `GammaUDD` instead interpolate
their stored secondary payloads.

Per-photon interpolation errors have distinct statuses. Coordinate
inversion, spatial-helper, and nonfinite spatial-output failures produce
`FAILURE_SPATIAL_INTERPOLATION`; temporal-window, temporal-helper, and
nonfinite final temporal-output failures produce
`FAILURE_TEMPORAL_INTERPOLATION`. The wrapper fills that ray's interpolation
scratch outputs with `NAN`. Numerical single- and batch-photon RKF45 finalization retry stage
interpolation failures with a halved step magnitude, bounded below by
`rkf45_h_min`, while preserving the last accepted state. Retryable trials use
`REJECTED`; exceeding `rkf45_max_retries` retains the interpolation-failure
status and ends that photon. Accepted-state metric-refresh failures retain
their failure status immediately. The temporal-stencil cap also bounds retry
magnitudes; incompatible limits produce `FAILURE_GENERIC`.
The batch integrator continues the remaining photons. Observer
interpolation remains fatal because no rays can be initialized without the
observer metric. Optional terminal and nonterminal normalization diagnostics
use temporary statuses, skip failed diagnostic samples, and leave their
sidecar entries as `NAN` without replacing a physical termination status.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/azimuthal_symmetry_spatial_lagrange_interpolation.py`, `temporal_lagrange_interpolation.py`, and `time_window_manager_numerical.py` — spatial interpolation, temporal interpolation, and time-window behavior; corroboration: `nrpy/infrastructures/BHaH/diagnostics/combine_raytracing_time_slices.py` — combined layout and metadata; `nrpy/infrastructures/BHaH/interpolation/differentiate_interpolation_lagrange_uniform.h` — uniform-basis derivative helper

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — interpolation failure statuses and synthetic-node tracking; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py` — `retry_interpolation_failures` and retry-step caps; `batch_integrator_numerical.py` — saved statuses and binary diagnostic output; `main_batch.py` — diagnostic filenames and parameters

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py` — single-photon tracker storage, evolution calls, and terminal reporting; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — synthetic-node detection and first-request recording

Numerical endpoint dispatch is piecewise constant. At or below the first stored
time, the first slice is spatially interpolated; at or above the selected final
slice, that final slice is reused. `g4DD` obtains temporal metric derivatives
from temporal interpolation. `g4DD_d0` uses stored metric derivatives on real
slices, but zeroes their temporal slots for direct endpoint dispatch and for
synthetic lower or upper nodes used to extend a frozen endpoint; the metric and
spatial-derivative slots remain copied from that endpoint. `GammaUDD` reuses
stored Christoffels at endpoints. The `g4DD_d0` padding policy is not controlled
by `--raytracing-static-christoffels`; that flag changes the exported final
Christoffel payload only.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py` — endpoint dispatch; `output_raytracing_data.py` — payload mode; corroboration: `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/temporal_lagrange_interpolation.py` — temporal endpoint handling

Startup reads `.bin` `NGHOSTS` metadata and uses that value to convert between
native coordinates and stored grid indices, including the two stored azimuthal
planes. The numerical reader does not use the finite-difference `NGHOSTS` macro
from `BHaH_defines.h` for these offsets. Startup terminates if the configured
spatial half-width requires more ghost zones than the `.bin` provides. It also
terminates when observer or positive Cartesian `r_escape` probes cannot support
the requested native stencil; the relevant `.par` controls are
`observer_x/y/z`, `r_escape`, and
`numerical_spacetime_spatial_interp_half_width`.

Evidence (public/scientific contract): `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py` — metadata loading and startup domain validation; `nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/azimuthal_symmetry_spatial_lagrange_interpolation.py` — stored grid indices; corroboration: `nrpy/infrastructures/BHaH/xx_tofrom_Cart.py` — Cartesian-to-native inversion and nearest stored index

The spatial helper `azimuthal_symmetry_spatial_lagrange_interpolation` supports
`SinhCylindricalv2n2` data with exactly two stored phi planes separated by pi.
The generated function rejects other plane counts or separations. It builds
`2*numerical_spacetime_spatial_interp_half_width + 1` stencil points in each
of the two non-azimuthal native directions around the selected center. The
center can be fixed by the caller for an RKF45 trial or selected from the
photon position.

The stencil reads the full stored grid, including ghost zones.
`azimuthal_symmetry_spatial_lagrange_point_index_from_full_payload_indices`
checks each storage index and converts it to the serialized point-record
index. Cartesian metric and connection components are accumulated directly
with the value weights; metric methods also use differentiated Lagrange
weights for the two native derivatives. The old reflected-node mapping and
per-node rotation are absent. After accumulation, the helper rotates the
interpolated tensors by `target_phi - phi_ref` to the target azimuth. Metric
methods obtain the phi derivative from that rotation and transform the native
spatial derivatives to Cartesian derivatives with the inverse reference-metric
Jacobian. This describes the generated calculation; it does not establish a
boundary-interpolation convergence rate.

Evidence (descriptive behavior): [azimuthal_symmetry_spatial_lagrange_interpolation.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/azimuthal_symmetry_spatial_lagrange_interpolation.py), `register_CFunction_azimuthal_symmetry_spatial_lagrange_interpolation`; corroboration: [output_raytracing_data.py](../../../nrpy/infrastructures/BHaH/diagnostics/output_raytracing_data.py), `register_CFunction_output_raytracing_data` (Cartesian tensor records including ghost zones)

Evolution-time raytracing export lives under BHaH diagnostics. When enabled by
`register_all_diagnostics`, the diagnostics function calls
`output_raytracing_data` on scheduled output steps. The exporter requires a
host/OpenMP build, `enable_rfm_precompute=True`,
`enable_RbarDD_gridfunctions=True`, one active grid, and the raytracing
example's `SinhCylindricalv2n2` source coordinate system. It refreshes
same-slice Ricci/RHS scratch data, evaluates Cartesian-basis `g4DD` and
`Gamma4UDD` symbolic recipes on interior points, fills the full logical-grid
payload including ghost zones, writes fixed-width metadata and binary64 point
records, and preserves the serialized logical-grid bounds. Optional
`--raytracing-static-christoffels` changes only the selected GammaUDD values in
the qualifying final output slice; interpolation consumes whichever values were
stored and applies its own endpoint policy. `combine_raytracing_time_slices.py`
then parses those stage-1 files, validates headers, sorts by simulation time,
and writes a read-only stacked container for downstream interpolation.

Evidence (descriptive behavior): `nrpy/infrastructures/BHaH/diagnostics/diagnostics.py` and `output_raytracing_data.py` — diagnostics registration and exporter; corroboration: `nrpy/infrastructures/BHaH/diagnostics/combine_raytracing_time_slices.py` — stage-1 validation and combined container

## Sources

- [puncture_diagnostics.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/puncture_diagnostics.py) - `puncture_diagnostics`, diagnostic readers, nearest-time association, and apparent-horizon containment
- [main_batch.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/main_batch.py) - `main`
- [batch_integrator_numerical.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_numerical.py) - `batch_integrator_numerical`
- [photon_batch_geodesic_integrator_numerical.py](../../../nrpy/examples/photon_batch_geodesic_integrator_numerical.py) - `--fast-light`, numerical interpolation and plane-refinement registration
- [batch_integrator_analytical.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/batch_integrator_analytical.py) - `batch_integrator_analytical`
- [single_integrator_analytical.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_analytical.py) - `single_integrator_analytical`
- [single_integrator_numerical.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/single_integrator_numerical.py) - `single_integrator_numerical`
- [normal_observer_log_energy.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/normal_observer_log_energy.py) - `normal_observer_log_energy`
- [set_initial_conditions_kernel.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/set_initial_conditions_kernel.py) - `set_initial_conditions_kernel`
- [time_slot_manager_helpers.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/time_slot_manager_helpers.py) - `time_slot_manager_helpers`, `TimeSlotManager`
- [interpolation_kernel.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/interpolation_kernel.py) - `interpolation_kernel`
- [calculate_ode_rhs_kernel.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/calculate_ode_rhs_kernel.py) - `calculate_ode_rhs_kernel`
- [rkf45_stage_update.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_stage_update.py) - `rkf45_stage_update`
- [rkf45_finalize_and_control_kernel.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/rkf45_finalize_and_control_kernel.py) - `rkf45_finalize_and_control_kernel`, `rkf45_finalize_and_control`
- [event_detection_manager_kernel.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/event_detection_manager_kernel.py) - `event_detection_manager_kernel`
- [find_plane_crossings_brent.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/find_plane_crossings_brent.py) - accepted crossing steps and analytical or numerical Brent refinement
- [handle_non_terminal_plane_intersection.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/handle_non_terminal_plane_intersection.py) - `handle_non_terminal_plane_intersection`
- [handle_terminal_plane_intersection.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/photon/handle_terminal_plane_intersection.py) - `handle_terminal_plane_intersection`
- [g4DD_metric.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/g4DD_metric.py) - `g4DD_metric`
- [connections.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/connections.py) - `connections`
- [ode_gsl_wrapper_massive.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/massive/ode_gsl_wrapper_massive.py) - `ode_gsl_wrapper_massive`
- [calculate_ode_rhs_massive.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/massive/calculate_ode_rhs_massive.py) - `calculate_ode_rhs_massive`
- [numerical_interpolation.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/numerical_interpolation.py) - `register_CFunction_numerical_interpolation`
- [time_window_manager_numerical.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/time_window_manager_numerical.py) - `time_window_manager_numerical`, `NumericalTimeWindowManager`, `time_window_manager_numerical_temporal_step_limit`, `time_window_manager_numerical_stencil_for_center`
- [azimuthal_symmetry_spatial_lagrange_interpolation.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/azimuthal_symmetry_spatial_lagrange_interpolation.py) - `register_CFunction_azimuthal_symmetry_spatial_lagrange_interpolation`
- [temporal_lagrange_interpolation.py](../../../nrpy/infrastructures/BHaH/general_relativity/geodesics/interpolation/temporal_lagrange_interpolation.py) - `register_CFunction_temporal_lagrange_interpolation`
- [differentiate_interpolation_lagrange_uniform.h](../../../nrpy/infrastructures/BHaH/interpolation/differentiate_interpolation_lagrange_uniform.h) - `compute_lagrange_basis_derivative_coeffs_xi`
- [xx_tofrom_Cart.py](../../../nrpy/infrastructures/BHaH/xx_tofrom_Cart.py) - `register_CFunction_Cart_to_xx_and_nearest_i0i1i2_assume_valid`
- [output_raytracing_data.py](../../../nrpy/infrastructures/BHaH/diagnostics/output_raytracing_data.py) - `register_CFunction_output_raytracing_data`, `raytracing_data_point_index_from_logical_indices`
- [combine_raytracing_time_slices.py](../../../nrpy/infrastructures/BHaH/diagnostics/combine_raytracing_time_slices.py) - `InputSliceInfo`, `Layout`
- [diagnostics.py](../../../nrpy/infrastructures/BHaH/diagnostics/diagnostics.py) - `register_all_diagnostics`, `enable_raytracing_data_output`

## See Also

- Parent: [BHaH](index.md)
- See also: [Diagnostics Output And Checkpointing](diagnostics-output-and-checkpointing.md)
- Depends on: [Geodesics](../../equations/general-relativity/geodesics.md)
- Example: [Black Hole Evolution](../../examples/black-hole-evolution.md)
- Depends on: [Grids, Coordinates, Reference Metrics, And Boundaries](grids-coordinates-reference-metrics-and-boundaries.md)

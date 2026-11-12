"""
Generate final-photon association and apparent-horizon containment diagnostics.

Saved BHaHAHA Cartesian rings define a radial surface about the solver's grid
origin, which can differ from the diagnostic centroid. Generated C helpers
recover that origin, interpolate the radial surface at a photon direction,
and write puncture IDs, containment bytes, and radial gaps at evolution-measure stops.

Author: OpenAI
"""

import nrpy.params as par


def puncture_diagnostics(include_batch_writer: bool = True) -> str:
    """
    Register the diagnostic-directory parameter and return local C helpers.

    :param include_batch_writer: Whether to include the combined batch blueprint writer.
    :return: C definitions for the numerical single or batch integrator.
    """
    par.register_CodeParameter(
        "char[4096]",
        __name__,
        "puncture_diagnostics_dir",
        "",
        commondata=True,
        add_to_parfile=True,
        description=(
            "Directory containing BHaHAHA_diagnostics.ah1.gp, .ah2.gp, .ah3.gp "
            "and h.tNNNNNNN.ahH.gp surfaces. Enables puncture-ID, horizon-status, "
            "and radial-gap records; an empty path disables this output."
        ),
    )
    helpers = r"""
    #include "interpolation/interpolation_lagrange_uniform.h"

    enum {
        PUNCTURE_UNCLASSIFIED = 0, PUNCTURE_1 = 1, PUNCTURE_2 = 2, MERGED_PUNCTURE = 3,
        HORIZON_OUTSIDE = 0, HORIZON_INSIDE = 1, HORIZON_NOT_CHECKED = 2
    }; // END ENUM: output byte values

    typedef struct {
        int ntheta, nphi;
        double origin[3], roundoff, pole_radius[2];
        double *radius;
    } puncture_surface_t; // END STRUCT: reconstructed radial horizon

    typedef struct {
        long iteration;
        double time, x, y, z;
        puncture_surface_t *surface;
    } puncture_position_t; // END STRUCT: saved horizon snapshot

    typedef struct {
        size_t count;
        puncture_position_t *positions;
    } puncture_track_t; // END STRUCT: horizon snapshot table

    typedef struct {
        puncture_track_t tracks[3];
        const char *directory;
        double common_horizon_time;
        size_t surfaces_loaded;
    } puncture_diagnostics_t; // END STRUCT: horizon diagnostic data

    typedef struct {
        uint8_t puncture_id, horizon_status;
        double horizon_time, radial_gap;
    } puncture_result_t; // END STRUCT: final photon association

    static void puncture_error(const char *path, const char *reason) {
        fprintf(stderr, "ERROR: puncture diagnostics '%s': %s.\n", path, reason);
        exit(EXIT_FAILURE);
    } // END FUNCTION: report diagnostic file error

    /**
     * Read saved iterations, times, and Cartesian horizon centroids.
     *
     * @param[in] directory Directory containing BHaHAHA diagnostic tables.
     * @param horizon Horizon number, from one to three.
     * @return Allocated rows with increasing times; the common table may be empty.
     */
    static puncture_track_t puncture_read_track(const char *directory, const int horizon) {
        char path[4096];
        const int length = snprintf(path, sizeof(path), "%s/BHaHAHA_diagnostics.ah%d.gp", directory, horizon);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            puncture_error(directory, "diagnostic path is too long");
        } // END IF: diagnostic path overflow
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            puncture_error(path, "cannot read diagnostic table");
        } // END IF: diagnostic open failed
        puncture_track_t track = {0};
        char line[8192];
        while (fgets(line, sizeof(line), file) != NULL) {
            char *row = line;
            while (isspace((unsigned char)*row)) {
                ++row;
            } // END WHILE: skip leading whitespace
            if (*row == '#' || *row == '\0') {
                continue;
            } // END IF: skip comment or blank
            puncture_position_t position = {0};
            if ((strchr(line, '\n') == NULL && !feof(file)) ||
                sscanf(row, "%ld %lf %lf %lf %lf", &position.iteration, &position.time,
                       &position.x, &position.y, &position.z) != 5 || position.iteration < 0 ||
                !isfinite(position.time) || !isfinite(position.x) ||
                !isfinite(position.y) || !isfinite(position.z) ||
                (track.count > 0 && position.time <= track.positions[track.count - 1].time)) {
                puncture_error(path, "invalid iteration, time, or centroid row");
            } // END IF: invalid diagnostic row
            if (track.count >= SIZE_MAX / sizeof(*track.positions)) {
                puncture_error(path, "too many diagnostic rows");
            } // END IF: position count overflow
            puncture_position_t *positions = realloc(track.positions, (track.count + 1) * sizeof(*positions));
            if (positions == NULL) {
                puncture_error(path, "cannot allocate diagnostic rows");
            } // END IF: position allocation failed
            track.positions = positions;
            track.positions[track.count++] = position;
        } // END WHILE: read horizon diagnostic rows
        const int read_failed = ferror(file);
        const int close_failed = fclose(file);
        if (read_failed || close_failed != 0) {
            puncture_error(path, "cannot finish reading diagnostic table");
        } // END IF: diagnostic read failed
        if (horizon != 3 && (track.count == 0 || track.positions[0].time != 0.0)) {
            puncture_error(path, "individual horizon table must begin at t=0");
        } // END IF: missing initial horizon position
        return track;
    } // END FUNCTION: read horizon snapshot table

    /**
     * Load the three diagnostic tables and determine first common-horizon time.
     *
     * @param[in] directory Diagnostic directory; an empty path disables output.
     * @return Diagnostic tables with surfaces initially unloaded.
     */
    static puncture_diagnostics_t puncture_load_diagnostics(const char *directory) {
        puncture_diagnostics_t data = {0};
        data.directory = directory;
        data.common_horizon_time = INFINITY;
        if (directory[0] != '\0') {
            for (int horizon = 0; horizon < 3; ++horizon) {
                data.tracks[horizon] = puncture_read_track(directory, horizon + 1);
            } // END LOOP: load horizon diagnostic tables
            if (data.tracks[2].count > 0) {
                data.common_horizon_time = data.tracks[2].positions[0].time;
            } // END IF: common horizon detected
        } // END IF: diagnostic output enabled
        return data;
    } // END FUNCTION: load horizon diagnostic tables

    /**
     * Select the nearest saved time, choosing the earlier row on a tie.
     *
     * @param[in,out] track Nonempty table whose surface pointer may be populated later.
     * @param time Requested nonnegative coordinate time.
     * @return Selected saved row.
     */
    static puncture_position_t *puncture_nearest_position(puncture_track_t *track, const double time) {
        size_t lower = 0, upper = track->count;
        while (lower < upper) {
            const size_t middle = lower + (upper - lower) / 2;
            if (track->positions[middle].time < time) {
                lower = middle + 1;
            } else {
                upper = middle;
            } // END ELSE: reduce upper time bound
        } // END WHILE: locate bracketing saved times
        if (lower == track->count || (lower > 0 &&
            time - track->positions[lower - 1].time <= track->positions[lower].time - time)) {
            --lower;
        } // END IF: select earlier nearest time
        return &track->positions[lower];
    } // END FUNCTION: select nearest saved snapshot

    /**
     * Recover the solver origin and radial grid from saved Cartesian rings.
     *
     * The cell-centered angular directions determine the origin through a
     * three-variable least-squares solve. Reuse a row's grid after its first load.
     *
     * @param[in,out] data Diagnostic directory and loaded-surface count.
     * @param horizon Horizon number selecting the surface filename.
     * @param[in,out] position Saved iteration and cached radial grid.
     * @return Validated positive radial surface about its reconstructed origin.
     */
    static puncture_surface_t *puncture_load_surface(puncture_diagnostics_t *data,
                                                    const int horizon, puncture_position_t *position) {
        if (position->surface != NULL) {
            return position->surface;
        } // END IF: reuse loaded radial surface
        char path[4096];
        const int length = snprintf(path, sizeof(path), "%s/h.t%07ld.ah%d.gp",
                                    data->directory, position->iteration, horizon);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            puncture_error(data->directory, "surface path is too long");
        } // END IF: surface path overflow
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            puncture_error(path, "cannot read expected horizon surface");
        } // END IF: surface open failed
        double (*points)[3] = NULL;
        size_t count = 0, capacity = 0, ring_size = 0, row_size = 0, rings = 0;
        double coordinate_scale = 0.0;
        char line[8192];
        while (1) {
            const bool at_end = fgets(line, sizeof(line), file) == NULL;
            char *row = line;
            if (!at_end) {
                while (isspace((unsigned char)*row)) {
                    ++row;
                } // END WHILE: skip leading whitespace
                if (*row == '#') {
                    continue;
                } // END IF: skip surface comment
            } // END IF: surface row available
            if (at_end || *row == '\0') {
                if (ring_size != 0) {
                    if (row_size == 0) {
                        row_size = ring_size;
                    } // END IF: first angular ring
                    if (ring_size != row_size) {
                        puncture_error(path, "angular rings have different lengths");
                    } // END IF: inconsistent angular ring
                    ++rings;
                    ring_size = 0;
                } // END IF: finish angular ring
                if (at_end) {
                    break;
                } // END IF: surface file ended
                continue;
            } // END IF: angular ring boundary
            if (count == capacity) {
                if (capacity > SIZE_MAX / 2) {
                    puncture_error(path, "too many surface points");
                } // END IF: point capacity overflow
                capacity = capacity == 0 ? 256 : 2 * capacity;
                if (capacity > SIZE_MAX / sizeof(*points)) {
                    puncture_error(path, "surface allocation size overflow");
                } // END IF: point allocation overflow
                double (*resized)[3] = realloc(points, capacity * sizeof(*points));
                if (resized == NULL) {
                    puncture_error(path, "cannot allocate surface points");
                } // END IF: point allocation failed
                points = resized;
            } // END IF: grow surface point array
            if ((strchr(line, '\n') == NULL && !feof(file)) ||
                sscanf(row, "%lf %lf %lf", &points[count][0], &points[count][1], &points[count][2]) != 3) {
                puncture_error(path, "invalid Cartesian surface row");
            } // END IF: invalid Cartesian row
            for (int axis = 0; axis < 3; ++axis) {
                if (!isfinite(points[count][axis])) {
                    puncture_error(path, "nonfinite surface coordinate");
                } // END IF: invalid surface coordinate
                coordinate_scale = fmax(coordinate_scale, fabs(points[count][axis]));
            } // END LOOP: check surface coordinate components
            ++count;
            ++ring_size;
        } // END WHILE: read Cartesian surface rings
        const int read_failed = ferror(file);
        const int close_failed = fclose(file);
        if (read_failed || close_failed != 0) {
            puncture_error(path, "cannot finish reading horizon surface");
        } // END IF: surface read failed
        if (rings < 2 || rings > INT_MAX || row_size < 3 || row_size - 1 > INT_MAX ||
            (row_size - 1) % 2 != 0) {
            puncture_error(path, "invalid cell-centered angular grid dimensions");
        } // END IF: invalid angular dimensions
        puncture_surface_t *surface = calloc(1, sizeof(*surface));
        if (surface == NULL) {
            puncture_error(path, "cannot allocate radial surface");
        } // END IF: surface allocation failed
        surface->ntheta = (int)rings;
        surface->nphi = (int)(row_size - 1);
        surface->roundoff = 1.0e-9 * coordinate_scale;

        // Solve sum(I-n*n^T)*origin = sum((I-n*n^T)*point).
        // This recovers the writer's grid origin without substituting its centroid.
        double system[3][4] = {{0}};
        for (int i = 0; i < surface->ntheta; ++i) {
            const double theta = (i + 0.5) * M_PI / surface->ntheta;
            for (int j = 0; j < surface->nphi; ++j) {
                const double phi = -M_PI + (j + 0.5) * 2.0 * M_PI / surface->nphi;
                const double n[3] = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};
                const double *p = points[(size_t)i * row_size + j];
                const double dot = n[0]*p[0] + n[1]*p[1] + n[2]*p[2];
                for (int a = 0; a < 3; ++a) {
                    system[a][3] += p[a] - n[a]*dot;
                    for (int b = 0; b < 3; ++b) {
                        system[a][b] += (a == b ? 1.0 : 0.0) - n[a]*n[b];
                    } // END LOOP: accumulate normal matrix row
                } // END LOOP: accumulate origin equations
            } // END LOOP: use distinct azimuthal points
        } // END LOOP: use all angular rings
        for (int column = 0; column < 3; ++column) {
            int pivot = column;
            for (int row = column + 1; row < 3; ++row) {
                if (fabs(system[row][column]) > fabs(system[pivot][column])) {
                    pivot = row;
                } // END IF: larger pivot available
            } // END LOOP: select elimination pivot
            if (fabs(system[pivot][column]) < 1.0e-12) {
                puncture_error(path, "cannot determine surface grid origin");
            } // END IF: singular origin equations
            for (int j = 0; j < 4; ++j) {
                const double tmp = system[column][j];
                system[column][j] = system[pivot][j];
                system[pivot][j] = tmp;
            } // END LOOP: swap elimination rows
            const double divisor = system[column][column];
            for (int j = column; j < 4; ++j) {
                system[column][j] /= divisor;
            } // END LOOP: normalize pivot equation
            for (int row = 0; row < 3; ++row) {
                if (row == column) {
                    continue;
                } // END IF: skip pivot row
                const double factor = system[row][column];
                for (int j = column; j < 4; ++j) {
                    system[row][j] -= factor * system[column][j];
                } // END LOOP: eliminate origin component
            } // END LOOP: update other origin equations
        } // END LOOP: solve three origin equations
        for (int a = 0; a < 3; ++a) {
            surface->origin[a] = system[a][3];
        } // END LOOP: save radial grid origin
        surface->radius = malloc(rings * (row_size - 1) * sizeof(double));
        if (surface->radius == NULL) {
            puncture_error(path, "cannot allocate radial grid");
        } // END IF: radial allocation failed
        for (int i = 0; i < surface->ntheta; ++i) {
            const double theta = (i + 0.5) * M_PI / surface->ntheta;
            for (int a = 0; a < 3; ++a) {
                if (fabs(points[(size_t)i*row_size][a] - points[(size_t)i*row_size + row_size-1][a]) > surface->roundoff) {
                    puncture_error(path, "angular ring closing point does not match");
                } // END IF: invalid periodic closing point
            } // END LOOP: check repeated closing point
            for (int j = 0; j < surface->nphi; ++j) {
                const double phi = -M_PI + (j + 0.5) * 2.0 * M_PI / surface->nphi;
                const double n[3] = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};
                double delta[3];
                for (int a = 0; a < 3; ++a) {
                    delta[a] = points[(size_t)i*row_size+j][a] - surface->origin[a];
                } // END LOOP: subtract radial grid origin
                const double radius = delta[0]*n[0] + delta[1]*n[1] + delta[2]*n[2];
                const double residual = hypot(hypot(delta[0]-radius*n[0], delta[1]-radius*n[1]), delta[2]-radius*n[2]);
                if (!isfinite(radius) || radius <= 0.0 || residual > surface->roundoff) {
                    puncture_error(path, "points do not match a positive radial surface");
                } // END IF: invalid radial representation
                surface->radius[(size_t)i*surface->nphi+j] = radius;
                if (i == 0) {
                    surface->pole_radius[0] += radius / surface->nphi;
                } // END IF: accumulate north pole estimate
                if (i == surface->ntheta - 1) {
                    surface->pole_radius[1] += radius / surface->nphi;
                } // END IF: accumulate south pole estimate
            } // END LOOP: recover distinct radial samples
        } // END LOOP: validate and recover radial grid
        free(points);
        position->surface = surface;
        ++data->surfaces_loaded;
        return surface;
    } // END FUNCTION: reconstruct saved radial horizon

    /**
     * Read one radial sample with periodic azimuth and polar reflection.
     *
     * @param[in] surface Loaded cell-centered radial grid.
     * @param i Polar index, including reflected neighbors.
     * @param j Azimuthal index, including periodic neighbors.
     * @return Radial sample after applying spherical angular boundaries.
     */
    static double puncture_radial_sample(const puncture_surface_t *surface, int i, int j) {
        if (i < 0) {
            i = -i - 1;
            j += surface->nphi / 2;
        } else if (i >= surface->ntheta) {
            i = 2*surface->ntheta - i - 1;
            j += surface->nphi / 2;
        } // END ELSE IF: reflect south polar neighbor
        j = (j % surface->nphi + surface->nphi) % surface->nphi;
        return surface->radius[(size_t)i*surface->nphi+j];
    } // END FUNCTION: apply spherical angular boundaries

    /**
     * Compare a Cartesian endpoint with a nearest-time apparent horizon.
     *
     * Quadratic Lagrange interpolation uses three samples per angular direction.
     * At a pole,
     * use the mean of the nearest polar ring; include surface-roundoff equality.
     *
     * @param[in,out] data Diagnostic tables and cached surfaces.
     * @param horizon Horizon number, from one to three.
     * @param time Nonnegative lookup time.
     * @param[in] point Finite Cartesian photon position.
     * @return Horizon ID, containment status, saved time, and signed radial gap.
     *
     * @note Unavailable checks and an undefined radial direction have NaN gaps.
     */
    static puncture_result_t puncture_check_horizon(puncture_diagnostics_t *data, const int horizon,
                                          const double time, const double point[3]) {
        puncture_result_t result = {(uint8_t)horizon, HORIZON_NOT_CHECKED, NAN, NAN};
        puncture_track_t *track = &data->tracks[horizon - 1];
        if (track->count == 0 || time < track->positions[0].time ||
            time > track->positions[track->count - 1].time) {
            return result;
        } // END IF: snapshot time coverage unavailable
        puncture_position_t *position = puncture_nearest_position(track, time);
        puncture_surface_t *surface = puncture_load_surface(data, horizon, position);
        result.horizon_time = position->time;
        const double dx = point[0]-surface->origin[0], dy = point[1]-surface->origin[1], dz = point[2]-surface->origin[2];
        const double rho = hypot(dx, dy), r = hypot(rho, dz);
        if (r <= surface->roundoff) {
            result.horizon_status = HORIZON_INSIDE;
            return result;
        } // END IF: photon at radial origin
        double radius;
        if (rho <= surface->roundoff) {
            radius = surface->pole_radius[dz >= 0.0 ? 0 : 1];
        } else {
            const double theta_index = acos(fmax(-1.0, fmin(1.0, dz/r))) * surface->ntheta/M_PI - 0.5;
            const double phi_index = (atan2(dy, dx)+M_PI) * surface->nphi/(2.0*M_PI) - 0.5;
            const int i = (int)floor(theta_index + 0.5) - 1;
            const int j = (int)floor(phi_index + 0.5) - 1;
            const REAL nodes[3] = {0.0, 1.0, 2.0};
            REAL inv_denom[3], theta_diffs[3], phi_diffs[3];
            REAL theta_weights[3], phi_weights[3];
            compute_inv_denom(3, inv_denom);
            compute_diffs_xi(3, theta_index-i, nodes, theta_diffs);
            compute_diffs_xi(3, phi_index-j, nodes, phi_diffs);
            compute_lagrange_basis_coeffs_xi(3, inv_denom, theta_diffs, theta_weights);
            compute_lagrange_basis_coeffs_xi(3, inv_denom, phi_diffs, phi_weights);
            radius = 0.0;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    radius += theta_weights[a]*phi_weights[b]*puncture_radial_sample(surface,i+a,j+b);
                } // END LOOP: sum azimuthal interpolation weights
            } // END LOOP: sum polar interpolation weights
        } // END ELSE: interpolate nonpolar surface direction
        result.radial_gap = r - radius;
        result.horizon_status = r <= radius + surface->roundoff ? HORIZON_INSIDE : HORIZON_OUTSIDE;
        return result;
    } // END FUNCTION: check nearest snapshot containment

    /**
     * Associate a final photon and check containment, nearer horizon first.
     *
     * Before common-horizon detection, try the second individual surface only
     * when the first does not establish containment. A containing second
     * surface replaces the nearest-centroid ID. After detection, test only the
     * common horizon. Negative times use the zero-time snapshot.
     *
     * @param[in,out] data Diagnostic tables and cached radial surfaces.
     * @param time Final photon time, or observer time for fast light.
     * @param[in] point Final Cartesian photon position.
     * @return Puncture ID, containment status, selected surface time, and radial gap.
     */
    static puncture_result_t puncture_classify_endpoint(puncture_diagnostics_t *data,
        double time, const double point[3]) {
        puncture_result_t result = {PUNCTURE_UNCLASSIFIED, HORIZON_NOT_CHECKED, NAN, NAN};
        if (!isfinite(time) || !isfinite(point[0]) || !isfinite(point[1]) || !isfinite(point[2])) {
            return result;
        } // END IF: invalid photon endpoint
        time = fmax(time, 0.0);
        if (time >= data->common_horizon_time) {
            return puncture_check_horizon(data, 3, time, point);
        } // END IF: common horizon epoch
        puncture_position_t *first = puncture_nearest_position(&data->tracks[0], time);
        puncture_position_t *second = puncture_nearest_position(&data->tracks[1], time);
        const double d1 = hypot(hypot(point[0]-first->x, point[1]-first->y), point[2]-first->z);
        const double d2 = hypot(hypot(point[0]-second->x, point[1]-second->y), point[2]-second->z);
        result.puncture_id = d1 < d2 ? PUNCTURE_1 : d2 < d1 ? PUNCTURE_2 : PUNCTURE_UNCLASSIFIED;
        const int nearer = d1 <= d2 ? 1 : 2;
        const puncture_result_t near_result = puncture_check_horizon(data, nearer, time, point);
        if (near_result.horizon_status == HORIZON_INSIDE) {
            return near_result;
        } // END IF: nearer horizon contains photon
        const int farther = 3-nearer;
        const puncture_result_t far_result = puncture_check_horizon(data, farther, time, point);
        if (far_result.horizon_status == HORIZON_INSIDE) {
            return far_result;
        } else if (near_result.horizon_status == HORIZON_OUTSIDE && far_result.horizon_status == HORIZON_OUTSIDE) {
            result.horizon_status = HORIZON_OUTSIDE;
            if (result.puncture_id != PUNCTURE_UNCLASSIFIED) {
                return near_result;
            } // END IF: unique nearest horizon selected
        } // END ELSE IF: both horizons exclude photon
        return result;
    } // END FUNCTION: classify final photon endpoint

    /**
     * Release every loaded surface and diagnostic table.
     *
     * @param[in,out] data Diagnostic allocations to release after blueprint output.
     */
    static void puncture_free_diagnostics(puncture_diagnostics_t *data) {
        for (int h = 0; h < 3; ++h) {
            for (size_t i = 0; i < data->tracks[h].count; ++i) {
                puncture_surface_t *surface = data->tracks[h].positions[i].surface;
                if (surface != NULL) {
                    free(surface->radius);
                    free(surface);
                } // END IF: release loaded radial surface
            } // END LOOP: release cached horizon snapshots
            free(data->tracks[h].positions);
        } // END LOOP: release three diagnostic tables
    } // END FUNCTION: release horizon diagnostic data
    """
    if include_batch_writer:
        helpers += r"""
    /**
     * Write ID/status bytes and a radial-gap double per photon.
     *
     * Only final evolution-measure records enter the reported percentages.
     * Inside fractions exclude unavailable checks. Other termination types
     * receive unclassified/not-checked bytes and a NaN gap in main record order.
     * Records contain no structure padding between the two bytes and double.
     *
     * @param[in,out] data Diagnostic tables and lazily loaded radial surfaces.
     * @param[in] commondata Observer time and runtime parameters.
     * @param[in] photons Final photon states in structure-of-arrays order.
     * @param[in] results Final saved termination records after terminal-plane selection.
     * @param num_rays Number of photons in this tile.
     * @param[in] filename Combined puncture blueprint output path.
     * @param fast_light Whether horizon lookup uses the fixed observer time.
     */
    static void write_puncture_diagnostics(puncture_diagnostics_t *data,
        const commondata_struct *commondata, const PhotonStateSoA *photons,
        const blueprint_data_t *results, const long num_rays,
        const char *filename, const bool fast_light) {
        if (data->directory[0] == '\0') {
            return;
        } // END IF: diagnostic output disabled
        FILE *file = fopen(filename, "wb");
        if (file == NULL) {
            puncture_error(filename, "cannot open combined puncture blueprint");
        } // END IF: output open failed
        long counts[4][3] = {{0}};
        long evolution_stops = 0;
        uint8_t records[4096][2 + sizeof(double)];
        for (long start = 0; start < num_rays; start += 4096) {
            const size_t count = (size_t)((num_rays-start < 4096) ? num_rays-start : 4096);
            for (size_t offset = 0; offset < count; ++offset) {
                const long ray = start + (long)offset;
                puncture_result_t result = {PUNCTURE_UNCLASSIFIED, HORIZON_NOT_CHECKED, NAN, NAN};
                if (results[ray].termination_type == STOP_CONDITION_EVOLUTION_MEASURE_EXCEEDED) {
                    const double point[3] = {photons->f[num_rays+ray], photons->f[2*num_rays+ray], photons->f[3*num_rays+ray]};
                    const double time = fast_light ? commondata->t_start : results[ray].t_f;
                    result = puncture_classify_endpoint(data, time, point);
                    ++counts[result.puncture_id][result.horizon_status];
                    ++evolution_stops;
                } // END IF: evolution measure stop selected
                records[offset][0] = result.puncture_id;
                records[offset][1] = result.horizon_status;
                memcpy(records[offset] + 2, &result.radial_gap, sizeof(result.radial_gap));
            } // END LOOP: classify ordered photon endpoints
            if (fwrite(records, sizeof(records[0]), count, file) != count) {
                puncture_error(filename, "cannot write combined puncture records");
            } // END IF: output write failed
        } // END LOOP: write bounded record chunks
        if (fclose(file) != 0) {
            puncture_error(filename, "cannot close combined puncture blueprint");
        } // END IF: output close failed
        printf("\nPUNCTURE / APPARENT-HORIZON DIAGNOSTICS (this tile)\n");
        printf("Evolution-measure stops: %ld\n", evolution_stops);
        if (evolution_stops > 0) {
            const char *names[4] = {"unclassified", "puncture_1", "puncture_2", "merged_puncture"};
            long total_inside = 0, total_outside = 0, total_unknown = 0;
            printf("%-15s %10s %10s %26s %12s\n", "Association", "Photons", "% of stops", "Inside / checked (%)", "Not checked");
            for (int id = 0; id < 4; ++id) {
                const long checked = counts[id][HORIZON_INSIDE] + counts[id][HORIZON_OUTSIDE];
                const long assigned = checked + counts[id][HORIZON_NOT_CHECKED];
                char fraction[80];
                if (checked > 0) {
                    snprintf(fraction, sizeof(fraction), "%ld/%ld (%.2f%%)", counts[id][HORIZON_INSIDE], checked,
                             100.0 * counts[id][HORIZON_INSIDE] / checked);
                } else {
                    snprintf(fraction, sizeof(fraction), "n/a (0 checked)");
                } // END ELSE: no known containment results
                printf("%-15s %10ld %9.2f%% %26s %12ld\n", names[id], assigned,
                       100.0*assigned/evolution_stops, fraction, counts[id][HORIZON_NOT_CHECKED]);
                total_inside += counts[id][HORIZON_INSIDE];
                total_outside += counts[id][HORIZON_OUTSIDE];
                total_unknown += counts[id][HORIZON_NOT_CHECKED];
            } // END LOOP: report four association categories
            printf("Totals: inside=%ld outside=%ld not_checked=%ld; inside percentages exclude not_checked.\n",
                   total_inside, total_outside, total_unknown);
        } // END IF: report evolution stop fractions
        printf("Wrote %ld puncture records (%zu bytes each) to %s; surfaces loaded: %zu.\n",
               num_rays, sizeof(records[0]), filename, data->surfaces_loaded);
    } // END FUNCTION: write combined records and summary

    """
    return helpers

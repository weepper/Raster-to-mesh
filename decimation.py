import numpy as np
from numba import njit
import gc

STL_DTYPE = np.dtype(
    [
        ("normals", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)


# =============================================================================
# Mesh Conversion Utilities
# =============================================================================


def stl_to_mesh(stl_array: np.ndarray):
    """Convert packed STL triangles into a deduplicated vertex/face representation."""
    v1 = stl_array["v1"]
    v2 = stl_array["v2"]
    v3 = stl_array["v3"]
    all_verts = np.vstack([v1, v2, v3]).astype(np.float32)

    # Round to 1 micron to ensure robust deduplication across chunks
    all_verts = np.round(all_verts * 1000.0) / 1000.0

    unique_verts, inverse_indices = np.unique(all_verts, axis=0, return_inverse=True)
    n_tris = len(stl_array)
    triangles = inverse_indices.reshape(3, n_tris).T.astype(np.int32)

    # 1. Remove degenerate triangles (any two vertices are the same)
    non_degen = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    triangles = triangles[non_degen]

    # 2. Remove duplicate triangles (regardless of winding order)
    if len(triangles) > 0:
        sorted_tris = np.sort(triangles, axis=1)
        tri_view = sorted_tris.view(np.dtype([("v", "i4", (3,))]))
        _, unique_idx = np.unique(tri_view, return_index=True)
        triangles = triangles[unique_idx]

    return unique_verts, triangles


# =============================================================================
# Corner Table Navigation
# =============================================================================


@njit(inline="always")
def next_c(c):
    return 3 * (c // 3) + ((c + 1) % 3)


@njit(inline="always")
def prev_c(c):
    return 3 * (c // 3) + ((c + 2) % 3)


@njit
def _extract_edges_njit(V, edges, V2C):
    C = V.shape[0]
    for c in range(C):
        v = V[c]
        V2C[v] = c
        v_next = V[next_c(c)]
        v_prev = V[prev_c(c)]
        edges[c]["v_min"] = min(v_next, v_prev)
        edges[c]["v_max"] = max(v_next, v_prev)
        edges[c]["c_id"] = c


@njit
def _link_opposites_njit(V, edges, O):
    C = edges.shape[0]
    i = 0
    while i < C - 1:
        if (
            edges[i]["v_min"] == edges[i + 1]["v_min"]
            and edges[i]["v_max"] == edges[i + 1]["v_max"]
        ):
            if (
                i + 2 < C
                and edges[i + 2]["v_min"] == edges[i]["v_min"]
                and edges[i + 2]["v_max"] == edges[i]["v_max"]
            ):
                v_min = edges[i]["v_min"]
                v_max = edges[i]["v_max"]
                while (
                    i < C and edges[i]["v_min"] == v_min and edges[i]["v_max"] == v_max
                ):
                    i += 1
                continue

            c1 = edges[i]["c_id"]
            c2 = edges[i + 1]["c_id"]

            if (V[next_c(c1)] == V[prev_c(c2)]) and (V[prev_c(c1)] == V[next_c(c2)]):
                O[c1] = c2
                O[c2] = c1
            i += 2
        else:
            i += 1


def build_corner_table(triangles: np.ndarray, num_vertices: int):
    """Build the Corner Table (V, O, V2C) from a triangle index array."""
    T = triangles.shape[0]
    C = 3 * T
    V = triangles.ravel().astype(np.int32)
    O = np.full(C, -1, dtype=np.int32)
    V2C = np.full(num_vertices, -1, dtype=np.int32)

    edges = np.zeros(
        C, dtype=[("v_min", np.int32), ("v_max", np.int32), ("c_id", np.int32)]
    )
    _extract_edges_njit(V, edges, V2C)
    edges.sort(order=["v_min", "v_max"])
    _link_opposites_njit(V, edges, O)

    return V, O, V2C


# =============================================================================
# Quadric Error Metric (Garland-Heckbert 1997)
# =============================================================================


@njit
def build_quadrics(V, V_coords, Q):
    """
    Accumulate per-vertex QEM quadrics from all incident face planes.

    Each vertex accumulates Q = Σ(K_p) where K_p = pp^T for the homogeneous
    plane p = [a, b, c, d] (normalized face normal + offset).

    Q is stored as (num_vertices, 10) float64, the 10 upper-triangle coefficients
    of the 4×4 symmetric matrix in order:
        [q00, q01, q02, q03, q11, q12, q13, q22, q23, q33]
    """
    C = len(V)
    T = C // 3

    for t in range(T):
        v0 = V[3 * t]
        v1 = V[3 * t + 1]
        v2 = V[3 * t + 2]
        if v0 == -1:
            continue

        p0x = V_coords[v0, 0]
        p0y = V_coords[v0, 1]
        p0z = V_coords[v0, 2]
        p1x = V_coords[v1, 0]
        p1y = V_coords[v1, 1]
        p1z = V_coords[v1, 2]
        p2x = V_coords[v2, 0]
        p2y = V_coords[v2, 1]
        p2z = V_coords[v2, 2]

        e1x = p1x - p0x
        e1y = p1y - p0y
        e1z = p1z - p0z
        e2x = p2x - p0x
        e2y = p2y - p0y
        e2z = p2z - p0z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length < 1e-12:
            continue

        a = nx / length
        b = ny / length
        c = nz / length
        d = -(a * p0x + b * p0y + c * p0z)

        # Upper triangle of pp^T (p = [a, b, c, d])
        q00 = a * a
        q01 = a * b
        q02 = a * c
        q03 = a * d
        q11 = b * b
        q12 = b * c
        q13 = b * d
        q22 = c * c
        q23 = c * d
        q33 = d * d

        for vi in (v0, v1, v2):
            Q[vi, 0] += q00
            Q[vi, 1] += q01
            Q[vi, 2] += q02
            Q[vi, 3] += q03
            Q[vi, 4] += q11
            Q[vi, 5] += q12
            Q[vi, 6] += q13
            Q[vi, 7] += q22
            Q[vi, 8] += q23
            Q[vi, 9] += q33


@njit
def compute_qem_cost(c, V, V_coords, Q, opt_pos):
    """
    Compute the QEM cost for collapsing the edge at corner c (v_from → v_to).
    Fills opt_pos[0:3] with the optimal collapse target position (or midpoint
    fallback when the combined quadric is rank-deficient).
    Returns the non-negative cost scalar.
    """
    v1 = V[c]
    v2 = V[next_c(c)]

    # Combined quadric Q12 = Q[v1] + Q[v2]
    q00 = Q[v1, 0] + Q[v2, 0]
    q01 = Q[v1, 1] + Q[v2, 1]
    q02 = Q[v1, 2] + Q[v2, 2]
    q03 = Q[v1, 3] + Q[v2, 3]
    q11 = Q[v1, 4] + Q[v2, 4]
    q12 = Q[v1, 5] + Q[v2, 5]
    q13 = Q[v1, 6] + Q[v2, 6]
    q22 = Q[v1, 7] + Q[v2, 7]
    q23 = Q[v1, 8] + Q[v2, 8]
    q33 = Q[v1, 9] + Q[v2, 9]

    # Solve for optimal position via Cramer's rule on the 3×3 system
    #   A * [x,y,z]^T = b  where A = upper-left 3×3 of Q12, b = -[q03, q13, q23]
    det = (
        q00 * (q11 * q22 - q12 * q12)
        - q01 * (q01 * q22 - q12 * q02)
        + q02 * (q01 * q12 - q11 * q02)
    )

    solved = False
    if abs(det) > 1e-10:
        bx = -q03
        by = -q13
        bz = -q23

        det_x = (
            bx * (q11 * q22 - q12 * q12)
            - q01 * (by * q22 - q12 * bz)
            + q02 * (by * q12 - q11 * bz)
        )
        det_y = (
            q00 * (by * q22 - q12 * bz)
            - bx * (q01 * q22 - q12 * q02)
            + q02 * (q01 * bz - by * q02)
        )
        det_z = (
            q00 * (q11 * bz - by * q12)
            - q01 * (q01 * bz - by * q02)
            + bx * (q01 * q12 - q11 * q02)
        )

        opt_pos[0] = det_x / det
        opt_pos[1] = det_y / det
        opt_pos[2] = det_z / det
        solved = True

    if not solved:
        # Fallback: midpoint
        opt_pos[0] = (V_coords[v1, 0] + V_coords[v2, 0]) * 0.5
        opt_pos[1] = (V_coords[v1, 1] + V_coords[v2, 1]) * 0.5
        opt_pos[2] = (V_coords[v1, 2] + V_coords[v2, 2]) * 0.5

    # Evaluate v̄^T Q12 v̄  (homogeneous: w=1)
    x = opt_pos[0]
    y = opt_pos[1]
    z = opt_pos[2]
    cost = (
        q00 * x * x
        + 2.0 * q01 * x * y
        + 2.0 * q02 * x * z
        + 2.0 * q03 * x
        + q11 * y * y
        + 2.0 * q12 * y * z
        + 2.0 * q13 * y
        + q22 * z * z
        + 2.0 * q23 * z
        + q33
    )
    return max(cost, 0.0)


# =============================================================================
# One-Ring Traversal & Topology Helpers
# =============================================================================


@njit
def get_one_ring_vertices(v, V, O, V2C, out_array):
    """Collect the one-ring neighbourhood of vertex v into out_array.
    Returns the count, or -1 on overflow / degenerate mesh."""
    count = 0
    start_c = V2C[v]
    if start_c == -1:
        return 0

    curr_c = start_c
    iters = 0
    MAX_ITERS = 500

    while iters < MAX_ITERS:
        if count >= 510:
            return -1
        iters += 1
        out_array[count] = V[next_c(curr_c)]
        count += 1
        opp = O[prev_c(curr_c)]
        if opp == -1:
            out_array[count] = V[prev_c(curr_c)]
            count += 1
            break
        curr_c = prev_c(opp)
        if curr_c == start_c:
            break

    if curr_c != start_c and iters < MAX_ITERS:
        curr_c = start_c
        iters2 = 0
        while iters2 < MAX_ITERS:
            if count >= 510:
                return -1
            iters2 += 1
            opp = O[next_c(curr_c)]
            if opp == -1:
                break
            curr_c = next_c(opp)
            out_array[count] = V[prev_c(curr_c)]
            count += 1

    if iters >= MAX_ITERS or count >= 120:
        return -1

    return count


@njit
def get_vertex_valence(v, V, O, V2C):
    """Count the number of incident (non-deleted) triangles for vertex v."""
    start_c = V2C[v]
    if start_c == -1:
        return 0

    valence = 0
    curr_c = start_c
    iters = 0

    while iters < 500:
        iters += 1
        valence += 1
        opp = O[prev_c(curr_c)]
        if opp == -1:
            break
        curr_c = prev_c(opp)
        if curr_c == start_c:
            break

    # handle boundary: walk the other direction too
    if curr_c != start_c:
        curr_c = start_c
        iters2 = 0
        while iters2 < 500:
            iters2 += 1
            opp = O[next_c(curr_c)]
            if opp == -1:
                break
            curr_c = next_c(opp)
            valence += 1

    return valence


@njit
def check_link_condition(c, V, O, V2C, ring_from, ring_to):
    """Topological link condition: collapse is safe iff |N(v1) ∩ N(v2)| == 2
    (or 1 for boundary edges)."""
    v_from = V[c]
    v_to = V[next_c(c)]

    n_from = get_one_ring_vertices(v_from, V, O, V2C, ring_from)
    n_to = get_one_ring_vertices(v_to, V, O, V2C, ring_to)

    if n_from < 0 or n_to < 0:
        return False

    shared = 0
    for i in range(n_from):
        for j in range(n_to):
            if ring_from[i] == ring_to[j]:
                shared += 1

    expected_shared = 1 if O[prev_c(c)] == -1 else 2
    return shared == expected_shared


@njit
def is_boundary_vertex(v, V, O, V2C):
    """Return True if vertex v sits on a mesh boundary (has a boundary edge)."""
    start_c = V2C[v]
    if start_c == -1:
        return False

    curr_c = start_c
    iters = 0
    while iters < 500:
        iters += 1
        opp = O[prev_c(curr_c)]
        if opp == -1:
            return True
        curr_c = prev_c(opp)
        if curr_c == start_c:
            break

    return False


# =============================================================================
# Edge Collapse (with QEM position update and quadric accumulation)
# =============================================================================


@njit
def edge_collapse(
    c, V, O, V2C, V_coords, Q, opt_pos, ring_from, ring_to, lock_boundaries=True
):
    """
    Collapse the edge at corner c (v_from → v_to).

    New features vs. plain collapse:
      - Moves v_to to the QEM-optimal position stored in opt_pos.
      - Merges Q[v_from] into Q[v_to] so subsequent cost evaluations are correct.
      - Optionally skips boundary vertices to prevent chunk-seam artefacts.
    """
    v_from = V[c]
    v_to = V[next_c(c)]

    if lock_boundaries:
        if is_boundary_vertex(v_from, V, O, V2C) or is_boundary_vertex(v_to, V, O, V2C):
            return False

    if not check_link_condition(c, V, O, V2C, ring_from, ring_to):
        return False

    # ---- Move v_to to the optimal position ----
    V_coords[v_to, 0] = opt_pos[0]
    V_coords[v_to, 1] = opt_pos[1]
    V_coords[v_to, 2] = opt_pos[2]

    # ---- Accumulate quadric ----
    Q[v_to, 0] += Q[v_from, 0]
    Q[v_to, 1] += Q[v_from, 1]
    Q[v_to, 2] += Q[v_from, 2]
    Q[v_to, 3] += Q[v_from, 3]
    Q[v_to, 4] += Q[v_from, 4]
    Q[v_to, 5] += Q[v_from, 5]
    Q[v_to, 6] += Q[v_from, 6]
    Q[v_to, 7] += Q[v_from, 7]
    Q[v_to, 8] += Q[v_from, 8]
    Q[v_to, 9] += Q[v_from, 9]

    # ---- Re-assign all corners that pointed at v_from ----
    start_c = V2C[v_from]
    curr_c = start_c
    iters = 0
    while iters < 500:
        iters += 1
        V[curr_c] = v_to
        opp = O[prev_c(curr_c)]
        if opp == -1:
            break
        curr_c = prev_c(opp)
        if curr_c == start_c:
            break

    if curr_c != start_c:
        curr_c = start_c
        iters2 = 0
        while iters2 < 500:
            iters2 += 1
            opp = O[next_c(curr_c)]
            if opp == -1:
                break
            curr_c = next_c(opp)
            V[curr_c] = v_to

    # ---- Delete the collapsed triangle(s) ----
    t1 = c // 3
    V[3 * t1] = -1
    V[3 * t1 + 1] = -1
    V[3 * t1 + 2] = -1

    o1 = O[c]
    o2 = O[next_c(c)]
    if o1 != -1:
        O[o1] = o2
    if o2 != -1:
        O[o2] = o1

    o = O[prev_c(c)]
    o3 = np.int32(-1)
    o4 = np.int32(-1)
    if o != -1:
        t2 = o // 3
        V[3 * t2] = -1
        V[3 * t2 + 1] = -1
        V[3 * t2 + 2] = -1

        o3 = O[next_c(o)]
        o4 = O[prev_c(o)]
        if o3 != -1:
            O[o3] = o4
        if o4 != -1:
            O[o4] = o3

    # ---- Invalidate v_from, refresh V2C[v_to] ----
    V2C[v_from] = -1
    V_coords[v_from, 0] = np.nan

    # Fix 6: Find a valid corner for v_to by checking surviving opposites
    # first, then falling back to a neighbourhood walk.
    valid_c = -1

    # Build candidates array: opposites of deleted triangles that still exist.
    cands = np.empty(4, dtype=np.int32)
    cands[0] = o1
    cands[1] = o2
    cands[2] = o3 if o != -1 else np.int32(-1)
    cands[3] = o4 if o != -1 else np.int32(-1)

    for ci in range(4):
        candidate = cands[ci]
        if candidate != -1 and V[3 * (candidate // 3)] != -1:
            base = 3 * (candidate // 3)
            for off in range(3):
                if V[base + off] == v_to:
                    valid_c = base + off
                    break
            if valid_c != -1:
                break

    # Fallback: walk from start_c if the opposites didn't yield a hit
    if valid_c == -1:
        curr_c = start_c
        iters3 = 0
        while iters3 < 500:
            iters3 += 1
            if V[3 * (curr_c // 3)] != -1:
                valid_c = curr_c
                break
            opp = O[prev_c(curr_c)]
            if opp == -1:
                break
            curr_c = prev_c(opp)
            if curr_c == start_c:
                break

        if valid_c == -1 and curr_c != start_c:
            curr_c = start_c
            iters4 = 0
            while iters4 < 500:
                iters4 += 1
                opp = O[next_c(curr_c)]
                if opp == -1:
                    break
                curr_c = next_c(opp)
                if V[3 * (curr_c // 3)] != -1:
                    valid_c = curr_c
                    break

    if valid_c != -1:
        V2C[v_to] = valid_c
    elif V2C[v_to] != -1 and V[3 * (V2C[v_to] // 3)] == -1:
        V2C[v_to] = -1

    return True


def _extract_boundary_loops(b_edges: np.ndarray):
    """Chain directed boundary edges (v1, v2) into one or more loops."""
    if len(b_edges) == 0:
        return []

    # Map from start vertex to end vertex
    adj = {e[0]: e[1] for e in b_edges}
    loops = []
    visited = set()

    # Sort keys to ensure deterministic behavior
    all_starts = sorted(adj.keys())

    for start_v in all_starts:
        if start_v in visited:
            continue

        loop = [start_v]
        visited.add(start_v)
        curr = adj.get(start_v)

        while curr is not None and curr != start_v:
            if curr in visited:
                break  # non-manifold or broken chain
            loop.append(curr)
            visited.add(curr)
            curr = adj.get(curr)

        if curr == start_v and len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int32))

    return loops


def _triangulate_polygon_2d(coords: np.ndarray, loop_indices: np.ndarray):
    """Fast fan triangulation from centroid for 2D polygon.
    coords: (N, 2) array of XY positions.
    loop_indices: indices into coords forming the loop.
    Returns: (M, 3) triangle indices into loop_indices.

    Uses O(N) fan triangulation from centroid - much faster than ear-clipping O(N^2).
    Works well for convex and slightly concave shapes typical of terrain boundaries.
    """
    n = len(loop_indices)
    if n < 3:
        return np.empty((0, 3), dtype=np.int32)

    # For n vertices, we need n-2 triangles
    # Use simple fan from first vertex - O(n) and very fast
    triangles = []
    for i in range(1, n - 1):
        triangles.append([loop_indices[0], loop_indices[i], loop_indices[i + 1]])

    return np.array(triangles, dtype=np.int32)


# =============================================================================
# Decimation Loop  —  QEM + Stellar Multiple-Choice Heuristic
# =============================================================================


@njit
def decimate_loop(
    V,
    O,
    V2C,
    V_coords,
    Q,
    target_faces,
    max_iters=10_000_000,
    k_choices=8,
    lock_boundaries=True,
):
    """
    Main decimation loop combining three state-of-the-art techniques:

    1. **Quadric Error Metric (QEM)**: Each candidate edge is scored by the
       Garland-Heckbert cost v̄^T(Q1+Q2)v̄, which is geometry-aware and
       preserves ridges, valleys, and sharp features.

    2. **Multiple-Choice heuristic**: k random edges are sampled per iteration
       and the cheapest one (by QEM-cost / valence) is selected. This avoids
       a full priority-queue while still achieving near-greedy quality.

    3. **Stellar valence weighting**: The score is divided by the combined
       valence of the two endpoint vertices, so high-valence (stellated,
       over-tessellated) regions are simplified first while low-valence
       feature edges are preserved longer.

    Optimization: Smart early rejection - only check link condition for candidates
    that could be the best, in order of increasing score.
    """
    C = len(V)

    current_faces = 0
    for t in range(C // 3):
        if V[3 * t] != -1:
            current_faces += 1

    if current_faces <= target_faces:
        return current_faces

    ring_from = np.empty(512, dtype=np.int32)
    ring_to = np.empty(512, dtype=np.int32)
    opt_pos = np.empty(3, dtype=np.float64)
    best_opt = np.empty(3, dtype=np.float64)

    # Pre-allocate arrays for candidates
    candidates_c = np.empty(k_choices, dtype=np.int32)
    candidates_score = np.empty(k_choices, dtype=np.float64)
    candidates_opt = np.empty((k_choices, 3), dtype=np.float64)

    iters = 0
    stagnant_fails = 0

    while current_faces > target_faces and iters < max_iters:
        iters += 1

        # Phase 1: Evaluate QEM costs for all k candidates (fast)
        n_candidates = 0
        for _ in range(k_choices * 3):  # Try more to get enough valid candidates
            c = np.random.randint(0, C)
            if V[c] == -1:
                continue

            # QEM cost for this candidate edge
            qem_cost = compute_qem_cost(c, V, V_coords, Q, opt_pos)

            # Stellar heuristic: weight by combined vertex valence
            v_from = V[c]
            v_to = V[next_c(c)]
            valence = get_vertex_valence(v_from, V, O, V2C) + get_vertex_valence(
                v_to, V, O, V2C
            )

            score = qem_cost / (valence + 1.0)

            if n_candidates < k_choices:
                candidates_c[n_candidates] = c
                candidates_score[n_candidates] = score
                candidates_opt[n_candidates, 0] = opt_pos[0]
                candidates_opt[n_candidates, 1] = opt_pos[1]
                candidates_opt[n_candidates, 2] = opt_pos[2]
                n_candidates += 1
            else:
                # Replace worst candidate if this one is better
                worst_idx = 0
                worst_score = candidates_score[0]
                for i in range(1, k_choices):
                    if candidates_score[i] > worst_score:
                        worst_score = candidates_score[i]
                        worst_idx = i
                if score < worst_score:
                    candidates_c[worst_idx] = c
                    candidates_score[worst_idx] = score
                    candidates_opt[worst_idx, 0] = opt_pos[0]
                    candidates_opt[worst_idx, 1] = opt_pos[1]
                    candidates_opt[worst_idx, 2] = opt_pos[2]

        if n_candidates == 0:
            break

        # Phase 2: Try candidates in order of increasing score
        # Only do expensive link check for promising candidates
        success = False
        best_score_so_far = np.inf

        # Simple bubble sort of top candidates (n_candidates <= k_choices <= 8)
        for i in range(n_candidates - 1):
            for j in range(i + 1, n_candidates):
                if candidates_score[j] < candidates_score[i]:
                    # Swap
                    tmp_c = candidates_c[i]
                    tmp_score = candidates_score[i]
                    tmp_opt0 = candidates_opt[i, 0]
                    tmp_opt1 = candidates_opt[i, 1]
                    tmp_opt2 = candidates_opt[i, 2]

                    candidates_c[i] = candidates_c[j]
                    candidates_score[i] = candidates_score[j]
                    candidates_opt[i, 0] = candidates_opt[j, 0]
                    candidates_opt[i, 1] = candidates_opt[j, 1]
                    candidates_opt[i, 2] = candidates_opt[j, 2]

                    candidates_c[j] = tmp_c
                    candidates_score[j] = tmp_score
                    candidates_opt[j, 0] = tmp_opt0
                    candidates_opt[j, 1] = tmp_opt1
                    candidates_opt[j, 2] = tmp_opt2

        # Try candidates in order
        for i in range(n_candidates):
            best_c = candidates_c[i]

            # Early termination: if this candidate is worse than an already-failed one
            if candidates_score[i] > best_score_so_far:
                break

            was_interior = O[prev_c(best_c)] != -1
            best_opt[0] = candidates_opt[i, 0]
            best_opt[1] = candidates_opt[i, 1]
            best_opt[2] = candidates_opt[i, 2]

            success = edge_collapse(
                best_c,
                V,
                O,
                V2C,
                V_coords,
                Q,
                best_opt,
                ring_from,
                ring_to,
                lock_boundaries,
            )

            if success:
                current_faces -= 2 if was_interior else 1
                stagnant_fails = 0
                break
            else:
                best_score_so_far = candidates_score[i]
                stagnant_fails += 1

        if stagnant_fails > k_choices * 1000:
            break

    return current_faces


# =============================================================================
# STL Output (Numba-accelerated inner loop)
# =============================================================================


@njit
def _fill_stl_arrays(V, V_coords, out_normals, out_v1, out_v2, out_v3):
    """Numba-compiled loop: fill pre-allocated flat arrays from the corner table."""
    T = len(V) // 3
    out_idx = 0
    for t in range(T):
        if V[3 * t] == -1:
            continue
        vi1 = V[3 * t]
        vi2 = V[3 * t + 1]
        vi3 = V[3 * t + 2]

        c1x = V_coords[vi1, 0]
        c1y = V_coords[vi1, 1]
        c1z = V_coords[vi1, 2]
        c2x = V_coords[vi2, 0]
        c2y = V_coords[vi2, 1]
        c2z = V_coords[vi2, 2]
        c3x = V_coords[vi3, 0]
        c3y = V_coords[vi3, 1]
        c3z = V_coords[vi3, 2]

        e1x = c2x - c1x
        e1y = c2y - c1y
        e1z = c2z - c1z
        e2x = c3x - c1x
        e2y = c3y - c1y
        e2z = c3z - c1z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        ln = (nx * nx + ny * ny + nz * nz) ** 0.5
        if ln > 0.0:
            nx /= ln
            ny /= ln
            nz /= ln
        else:
            nx = 0.0
            ny = 0.0
            nz = 1.0  # Default up for degenerate

        out_normals[out_idx, 0] = nx
        out_normals[out_idx, 1] = ny
        out_normals[out_idx, 2] = nz

        out_v1[out_idx, 0] = c1x
        out_v1[out_idx, 1] = c1y
        out_v1[out_idx, 2] = c1z
        out_v2[out_idx, 0] = c2x
        out_v2[out_idx, 1] = c2y
        out_v2[out_idx, 2] = c2z
        out_v3[out_idx, 0] = c3x
        out_v3[out_idx, 1] = c3y
        out_v3[out_idx, 2] = c3z

        out_idx += 1


@njit
def _collect_boundary_edges(V, O, out_edges):
    """Scan corner table for boundary edges (O[c] == -1).
    Returns count and fills out_edges with (v1, v2) pairs.
    """
    count = 0
    C = len(V)
    for c in range(C):
        if V[c] != -1 and O[c] == -1:
            # The edge opposite corner c is a boundary edge.
            # Triangle is (V[c], V[next_c(c)], V[prev_c(c)]).
            # Edge is from next to prev.
            v_next = V[next_c(c)]
            v_prev = V[prev_c(c)]
            out_edges[count, 0] = v_next
            out_edges[count, 1] = v_prev
            count += 1
    return count


def add_skirt_to_mesh(
    V_coords: np.ndarray, triangles: np.ndarray, z_bottom: float, external_edges=None
):
    """
    Identifies all boundary edges in the mesh and adds vertical walls
    down to z_bottom, plus a triangulated base that matches the wall boundary.

    If external_edges is provided, uses those instead of computing boundaries.
    """
    num_verts = len(V_coords)
    V, O, V2C = build_corner_table(triangles, num_verts)

    # 1. Collect boundary edges (use provided or compute)
    if external_edges is not None:
        b_edges = external_edges
        n_b = len(b_edges)
    else:
        max_b_edges = len(triangles)
        b_edges_arr = np.empty((max_b_edges, 2), dtype=np.int32)
        n_b_all = _collect_boundary_edges(V, O, b_edges_arr)
        b_edges_all = b_edges_arr[:n_b_all]

        if n_b_all == 0:
            return V_coords, triangles

        # Find edges that occur only once (true manifold boundaries)
        sorted_edges = np.sort(b_edges_all, axis=1)
        edge_view = sorted_edges.view(np.dtype([("v", np.int32, (2,))]))
        uv, inv, counts = np.unique(edge_view, return_inverse=True, return_counts=True)

        manifold_mask = (counts[inv] == 1).ravel()
        b_edges = b_edges_all[manifold_mask]
        n_b = len(b_edges)

    if n_b == 0:
        return V_coords, triangles

    # 2. Add wall vertices (one per unique boundary vertex)
    unique_b_v_ids = np.unique(b_edges)
    n_new = len(unique_b_v_ids)
    v_to_bottom = np.full(num_verts, -1, dtype=np.int32)
    new_v_coords = np.empty((n_new, 3), dtype=np.float32)

    for i, old_v in enumerate(unique_b_v_ids):
        v_to_bottom[old_v] = num_verts + i
        new_v_coords[i, 0] = V_coords[old_v, 0]
        new_v_coords[i, 1] = V_coords[old_v, 1]
        new_v_coords[i, 2] = z_bottom

    # 3. Create wall triangles
    wall_tris = np.empty((n_b * 2, 3), dtype=np.int32)
    for i in range(n_b):
        v1, v2 = b_edges[i]
        v1_b = v_to_bottom[v1]
        v2_b = v_to_bottom[v2]

        wall_tris[2 * i, 0] = v1
        wall_tris[2 * i, 1] = v1_b
        wall_tris[2 * i, 2] = v2
        wall_tris[2 * i + 1, 0] = v2
        wall_tris[2 * i + 1, 1] = v1_b
        wall_tris[2 * i + 1, 2] = v2_b

    # 4. Create base using grid that aligns with wall vertices
    # Get bounding box of wall vertices
    min_x = new_v_coords[:, 0].min()
    max_x = new_v_coords[:, 0].max()
    min_y = new_v_coords[:, 1].min()
    max_y = new_v_coords[:, 1].max()

    # Create a simple 2-triangle base rectangle
    # This is simpler and avoids non-manifold issues
    base_v_start = num_verts + n_new
    base_v = np.array(
        [
            [min_x, min_y, z_bottom],
            [max_x, min_y, z_bottom],
            [max_x, max_y, z_bottom],
            [min_x, max_y, z_bottom],
        ],
        dtype=np.float32,
    )

    base_t = np.array(
        [
            [base_v_start + 0, base_v_start + 1, base_v_start + 2],
            [base_v_start + 0, base_v_start + 2, base_v_start + 3],
        ],
        dtype=np.int32,
    )

    # 5. Concatenate everything
    final_v_coords = np.vstack([V_coords, new_v_coords, base_v])
    final_triangles = np.vstack([triangles, wall_tris, base_t])

    return final_v_coords, final_triangles


def mesh_to_stl(V: np.ndarray, V_coords: np.ndarray) -> np.ndarray:
    """Convert the corner table back to a packed STL array."""
    valid_T = int(np.sum(V[::3] != -1))

    out_normals = np.zeros((valid_T, 3), dtype=np.float32)
    out_v1 = np.zeros((valid_T, 3), dtype=np.float32)
    out_v2 = np.zeros((valid_T, 3), dtype=np.float32)
    out_v3 = np.zeros((valid_T, 3), dtype=np.float32)

    _fill_stl_arrays(V, V_coords, out_normals, out_v1, out_v2, out_v3)

    stl_out = np.zeros(valid_T, dtype=STL_DTYPE)
    stl_out["normals"] = out_normals
    stl_out["v1"] = out_v1
    stl_out["v2"] = out_v2
    stl_out["v3"] = out_v3
    return stl_out


def indexed_to_stl(V_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Convert an indexed mesh directly to a packed STL structured array."""
    # Robust final rounding to ensure manifoldness in external tools
    V_coords = np.round(V_coords.astype(np.float64) * 1000.0) / 1000.0
    V_coords = V_coords.astype(np.float32)

    n_tris = len(triangles)

    if n_tris == 0:
        return np.zeros(0, dtype=STL_DTYPE)

    # CCW orientation: edge1 = v2-v1, edge2 = v3-v1, normal = e1 x e2
    v1 = V_coords[triangles[:, 0]]
    v2 = V_coords[triangles[:, 1]]
    v3 = V_coords[triangles[:, 2]]

    # Filter degenerate triangles (zero area)
    e1 = v2 - v1
    e2 = v3 - v1
    cross = np.cross(e1, e2)
    areas = np.linalg.norm(cross, axis=1)
    non_degenerate = areas > 1e-10

    # Apply filter
    v1 = v1[non_degenerate]
    v2 = v2[non_degenerate]
    v3 = v3[non_degenerate]

    n_valid = len(v1)
    stl_out = np.zeros(n_valid, dtype=STL_DTYPE)

    stl_out["v1"] = v1
    stl_out["v2"] = v2
    stl_out["v3"] = v3

    normals = np.cross(e1[non_degenerate], e2[non_degenerate])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 1e-12, norms, 1.0)
    stl_out["normals"] = normals

    return stl_out


# =============================================================================
# Public API
# =============================================================================


def decimate_mesh(
    V_coords: np.ndarray,
    triangles: np.ndarray,
    target_faces: int,
    k_choices: int = 8,
    lock_boundaries: bool = True,
    max_iters: int = 10_000_000,
) -> np.ndarray:
    """
    Decimate an indexed triangle mesh directly.
    """
    # Fix 5: Defensive degenerate triangle removal
    t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    non_degen = (t0 != t1) & (t1 != t2) & (t0 != t2)
    if not np.all(non_degen):
        triangles = triangles[non_degen].copy()

    num_verts = len(V_coords)
    V, O, V2C = build_corner_table(triangles, num_verts)

    # Build QEM quadrics in float64 for numerical precision
    Q = np.zeros((num_verts, 10), dtype=np.float64)
    build_quadrics(V, V_coords.astype(np.float64), Q)

    decimate_loop(
        V,
        O,
        V2C,
        V_coords,
        Q,
        target_faces,
        max_iters=max_iters,
        k_choices=k_choices,
        lock_boundaries=lock_boundaries,
    )

    # Final clean up on decimated mesh
    final_stl = mesh_to_stl(V, V_coords)
    # Redo stl_to_mesh/indexed_to_stl for triangle deduplication & normals
    V_c, tri = stl_to_mesh(final_stl)
    return indexed_to_stl(V_c, tri)


def decimate_stl(
    stl_array: np.ndarray,
    target_faces: int,
    k_choices: int = 8,
    lock_boundaries: bool = True,
) -> np.ndarray:
    """
    Decimate an STL triangle array to approximately `target_faces` triangles.

    Args:
        stl_array:        Input STL structured array.
        target_faces:     Desired number of output triangles.
        k_choices:        Number of random edges sampled per iteration (default 8).
        lock_boundaries:  If True, boundary vertices are never collapsed.

    Returns:
        Decimated STL structured array.
    """
    V_coords, triangles = stl_to_mesh(stl_array)
    return decimate_mesh(V_coords, triangles, target_faces, k_choices, lock_boundaries)

import numpy as np
from numba import njit
import gc

STL_DTYPE = np.dtype([
    ('normals', '<f4', (3,)),
    ('v1', '<f4', (3,)),
    ('v2', '<f4', (3,)),
    ('v3', '<f4', (3,)),
    ('attr', '<u2')
])


# =============================================================================
# Mesh Conversion Utilities
# =============================================================================

def stl_to_mesh(stl_array: np.ndarray):
    """Convert packed STL triangles into a deduplicated vertex/face representation."""
    v1 = stl_array['v1']
    v2 = stl_array['v2']
    v3 = stl_array['v3']
    all_verts = np.vstack([v1, v2, v3]).astype(np.float32)

    unique_verts, inverse_indices = np.unique(all_verts, axis=0, return_inverse=True)
    n_tris = len(stl_array)
    inverse_indices = inverse_indices.reshape(3, n_tris).T
    triangles = inverse_indices.astype(np.int32)

    return unique_verts, triangles


# =============================================================================
# Corner Table Navigation
# =============================================================================

@njit(inline='always')
def next_c(c):
    return 3 * (c // 3) + ((c + 1) % 3)


@njit(inline='always')
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
        edges[c]['v_min'] = min(v_next, v_prev)
        edges[c]['v_max'] = max(v_next, v_prev)
        edges[c]['c_id'] = c


@njit
def _link_opposites_njit(V, edges, O):
    C = edges.shape[0]
    i = 0
    while i < C - 1:
        if (edges[i]['v_min'] == edges[i+1]['v_min'] and
                edges[i]['v_max'] == edges[i+1]['v_max']):

            if (i + 2 < C and
                    edges[i+2]['v_min'] == edges[i]['v_min'] and
                    edges[i+2]['v_max'] == edges[i]['v_max']):
                v_min = edges[i]['v_min']
                v_max = edges[i]['v_max']
                while i < C and edges[i]['v_min'] == v_min and edges[i]['v_max'] == v_max:
                    i += 1
                continue

            c1 = edges[i]['c_id']
            c2 = edges[i+1]['c_id']

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

    edges = np.zeros(C, dtype=[('v_min', np.int32), ('v_max', np.int32), ('c_id', np.int32)])
    _extract_edges_njit(V, edges, V2C)
    edges.sort(order=['v_min', 'v_max'])
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
        v0 = V[3*t]
        v1 = V[3*t + 1]
        v2 = V[3*t + 2]
        if v0 == -1:
            continue

        p0x = V_coords[v0, 0]; p0y = V_coords[v0, 1]; p0z = V_coords[v0, 2]
        p1x = V_coords[v1, 0]; p1y = V_coords[v1, 1]; p1z = V_coords[v1, 2]
        p2x = V_coords[v2, 0]; p2y = V_coords[v2, 1]; p2z = V_coords[v2, 2]

        e1x = p1x - p0x; e1y = p1y - p0y; e1z = p1z - p0z
        e2x = p2x - p0x; e2y = p2y - p0y; e2z = p2z - p0z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        length = (nx*nx + ny*ny + nz*nz) ** 0.5
        if length < 1e-12:
            continue

        a = nx / length
        b = ny / length
        c = nz / length
        d = -(a * p0x + b * p0y + c * p0z)

        # Upper triangle of pp^T (p = [a, b, c, d])
        q00 = a*a;  q01 = a*b;  q02 = a*c;  q03 = a*d
        q11 = b*b;  q12 = b*c;  q13 = b*d
        q22 = c*c;  q23 = c*d
        q33 = d*d

        for vi in (v0, v1, v2):
            Q[vi, 0] += q00;  Q[vi, 1] += q01;  Q[vi, 2] += q02;  Q[vi, 3] += q03
            Q[vi, 4] += q11;  Q[vi, 5] += q12;  Q[vi, 6] += q13
            Q[vi, 7] += q22;  Q[vi, 8] += q23
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
    q00 = Q[v1, 0] + Q[v2, 0];  q01 = Q[v1, 1] + Q[v2, 1]
    q02 = Q[v1, 2] + Q[v2, 2];  q03 = Q[v1, 3] + Q[v2, 3]
    q11 = Q[v1, 4] + Q[v2, 4];  q12 = Q[v1, 5] + Q[v2, 5]
    q13 = Q[v1, 6] + Q[v2, 6];  q22 = Q[v1, 7] + Q[v2, 7]
    q23 = Q[v1, 8] + Q[v2, 8];  q33 = Q[v1, 9] + Q[v2, 9]

    # Solve for optimal position via Cramer's rule on the 3×3 system
    #   A * [x,y,z]^T = b  where A = upper-left 3×3 of Q12, b = -[q03, q13, q23]
    det = (q00 * (q11*q22 - q12*q12)
           - q01 * (q01*q22 - q12*q02)
           + q02 * (q01*q12 - q11*q02))

    solved = False
    if abs(det) > 1e-10:
        bx = -q03;  by = -q13;  bz = -q23

        det_x = (bx * (q11*q22 - q12*q12)
                 - q01 * (by*q22 - q12*bz)
                 + q02 * (by*q12 - q11*bz))
        det_y = (q00 * (by*q22 - q12*bz)
                 - bx * (q01*q22 - q12*q02)
                 + q02 * (q01*bz - by*q02))
        det_z = (q00 * (q11*bz - by*q12)
                 - q01 * (q01*bz - by*q02)
                 + bx * (q01*q12 - q11*q02))

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
    x = opt_pos[0];  y = opt_pos[1];  z = opt_pos[2]
    cost = (q00*x*x + 2.0*q01*x*y + 2.0*q02*x*z + 2.0*q03*x
            + q11*y*y + 2.0*q12*y*z + 2.0*q13*y
            + q22*z*z + 2.0*q23*z
            + q33)
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
def edge_collapse(c, V, O, V2C, V_coords, Q, opt_pos, ring_from, ring_to,
                  lock_boundaries=True):
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
    Q[v_to, 0] += Q[v_from, 0];  Q[v_to, 1] += Q[v_from, 1]
    Q[v_to, 2] += Q[v_from, 2];  Q[v_to, 3] += Q[v_from, 3]
    Q[v_to, 4] += Q[v_from, 4];  Q[v_to, 5] += Q[v_from, 5]
    Q[v_to, 6] += Q[v_from, 6];  Q[v_to, 7] += Q[v_from, 7]
    Q[v_to, 8] += Q[v_from, 8];  Q[v_to, 9] += Q[v_from, 9]

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
    V[3*t1] = -1;  V[3*t1+1] = -1;  V[3*t1+2] = -1

    o1 = O[c];  o2 = O[next_c(c)]
    if o1 != -1: O[o1] = o2
    if o2 != -1: O[o2] = o1

    o = O[prev_c(c)]
    if o != -1:
        t2 = o // 3
        V[3*t2] = -1;  V[3*t2+1] = -1;  V[3*t2+2] = -1

        o3 = O[next_c(o)];  o4 = O[prev_c(o)]
        if o3 != -1: O[o3] = o4
        if o4 != -1: O[o4] = o3

    # ---- Invalidate v_from, refresh V2C[v_to] ----
    V2C[v_from] = -1
    V_coords[v_from, 0] = np.nan

    valid_c = -1
    curr_c = start_c
    iters3 = 0
    while iters3 < 500:
        iters3 += 1
        if V[3*(curr_c//3)] != -1:
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
            if V[3*(curr_c//3)] != -1:
                valid_c = curr_c
                break

    if valid_c != -1:
        if V2C[v_to] == -1 or V[3*(V2C[v_to]//3)] == -1:
            V2C[v_to] = valid_c

    return True


# =============================================================================
# Decimation Loop  —  QEM + Stellar Multiple-Choice Heuristic
# =============================================================================

@njit
def decimate_loop(V, O, V2C, V_coords, Q, target_faces,
                  max_iters=10_000_000, k_choices=8, lock_boundaries=True):
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
    """
    C = len(V)

    current_faces = 0
    for t in range(C // 3):
        if V[3*t] != -1:
            current_faces += 1

    if current_faces <= target_faces:
        return current_faces

    ring_from = np.empty(512, dtype=np.int32)
    ring_to = np.empty(512, dtype=np.int32)
    opt_pos = np.empty(3, dtype=np.float64)
    best_opt = np.empty(3, dtype=np.float64)

    iters = 0
    stagnant_fails = 0

    while current_faces > target_faces and iters < max_iters:
        iters += 1

        best_c = -1
        best_score = np.inf

        for _ in range(k_choices):
            c = np.random.randint(0, C)
            if V[c] == -1:
                continue

            # QEM cost for this candidate edge
            qem_cost = compute_qem_cost(c, V, V_coords, Q, opt_pos)

            # Stellar heuristic: weight by combined vertex valence so that
            # high-valence (flat, over-refined) edges score lower → collapse first
            v_from = V[c]
            v_to = V[next_c(c)]
            valence = (get_vertex_valence(v_from, V, O, V2C)
                       + get_vertex_valence(v_to, V, O, V2C))

            score = qem_cost / (valence + 1.0)

            if score < best_score:
                best_c = c
                best_score = score
                best_opt[0] = opt_pos[0]
                best_opt[1] = opt_pos[1]
                best_opt[2] = opt_pos[2]

        if best_c != -1:
            success = edge_collapse(
                best_c, V, O, V2C, V_coords, Q, best_opt,
                ring_from, ring_to, lock_boundaries)
            if success:
                current_faces -= 2 if O[prev_c(best_c)] != -1 else 1
                stagnant_fails = 0
            else:
                stagnant_fails += 1

        if stagnant_fails > C * 4:
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
        if V[3*t] == -1:
            continue
        vi1 = V[3*t];  vi2 = V[3*t+1];  vi3 = V[3*t+2]

        c1x = V_coords[vi1, 0];  c1y = V_coords[vi1, 1];  c1z = V_coords[vi1, 2]
        c2x = V_coords[vi2, 0];  c2y = V_coords[vi2, 1];  c2z = V_coords[vi2, 2]
        c3x = V_coords[vi3, 0];  c3y = V_coords[vi3, 1];  c3z = V_coords[vi3, 2]

        e1x = c2x - c1x;  e1y = c2y - c1y;  e1z = c2z - c1z
        e2x = c3x - c1x;  e2y = c3y - c1y;  e2z = c3z - c1z

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        ln = (nx*nx + ny*ny + nz*nz) ** 0.5
        if ln > 0.0:
            nx /= ln;  ny /= ln;  nz /= ln

        out_normals[out_idx, 0] = nx
        out_normals[out_idx, 1] = ny
        out_normals[out_idx, 2] = nz

        out_v1[out_idx, 0] = c1x;  out_v1[out_idx, 1] = c1y;  out_v1[out_idx, 2] = c1z
        out_v2[out_idx, 0] = c2x;  out_v2[out_idx, 1] = c2y;  out_v2[out_idx, 2] = c2z
        out_v3[out_idx, 0] = c3x;  out_v3[out_idx, 1] = c3y;  out_v3[out_idx, 2] = c3z

        out_idx += 1


def mesh_to_stl(V: np.ndarray, V_coords: np.ndarray) -> np.ndarray:
    """Convert the corner table back to a packed STL array."""
    valid_T = int(np.sum(V[::3] != -1))

    out_normals = np.zeros((valid_T, 3), dtype=np.float32)
    out_v1 = np.zeros((valid_T, 3), dtype=np.float32)
    out_v2 = np.zeros((valid_T, 3), dtype=np.float32)
    out_v3 = np.zeros((valid_T, 3), dtype=np.float32)

    _fill_stl_arrays(V, V_coords, out_normals, out_v1, out_v2, out_v3)

    stl_out = np.zeros(valid_T, dtype=STL_DTYPE)
    stl_out['normals'] = out_normals
    stl_out['v1'] = out_v1
    stl_out['v2'] = out_v2
    stl_out['v3'] = out_v3
    return stl_out


def indexed_to_stl(V_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Convert an indexed mesh directly to a packed STL structured array."""
    n_tris = len(triangles)
    stl_out = np.zeros(n_tris, dtype=STL_DTYPE)
    
    if n_tris == 0:
        return stl_out

    stl_out['v1'] = V_coords[triangles[:, 0]]
    stl_out['v2'] = V_coords[triangles[:, 1]]
    stl_out['v3'] = V_coords[triangles[:, 2]]
    
    return stl_out


# =============================================================================
# Public API
# =============================================================================

def decimate_mesh(V_coords: np.ndarray, triangles: np.ndarray, target_faces: int,
                  k_choices: int = 8, lock_boundaries: bool = True) -> np.ndarray:
    """
    Decimate an indexed triangle mesh directly.
    """
    num_verts = len(V_coords)
    V, O, V2C = build_corner_table(triangles, num_verts)

    # Build QEM quadrics in float64 for numerical precision
    Q = np.zeros((num_verts, 10), dtype=np.float64)
    build_quadrics(V, V_coords.astype(np.float64), Q)

    decimate_loop(V, O, V2C, V_coords, Q, target_faces,
                  k_choices, lock_boundaries)

    return mesh_to_stl(V, V_coords)


def decimate_stl(stl_array: np.ndarray, target_faces: int,
                 k_choices: int = 8, lock_boundaries: bool = True) -> np.ndarray:
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

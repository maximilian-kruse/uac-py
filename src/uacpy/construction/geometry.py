"""Stateless mesh-geometry algorithms behind the construction pipeline: boundary, path and region
extraction (igraph, trimesh, numba), and arclength / (alpha, beta) parameterization of paths.

Each function takes plain arrays/pyvista meshes in and returns plain arrays out -- no pipeline
state, so every one is independently testable. Vertex indices throughout are into the *original*
mesh unless a docstring says otherwise; paths are 1D arrays of such indices, ordered along the path.
"""

import igraph as ig
import numpy as np
import pyvista as pv
import trimesh as tm
from numba import njit
from numba.typed import List

from uacpy.construction.model import ParameterizedPath, Submesh, UACPath
from uacpy.output import logging


# ==================================================================================================
def get_feature_boundary(
    mesh: pv.PolyData,
    feature_tag: int,
    coincides_with_geometry_boundary: bool,
    logger: logging.BaseLogger | None = None,
) -> np.ndarray:
    """Extract the ordered vertex loop bounding the region tagged `feature_tag`.

    A tagged region's own boundary edges include both its "true" anatomical boundary (e.g. a PV
    ostium's rim) and, when the tag reaches the mesh's outer edge (`coincides_with_geometry_
    boundary=True`, e.g. the mitral valve annulus), the mesh's own cut boundary running along the
    same vertices. `coincides_with_geometry_boundary` says which of those this call wants:
    intersect with the mesh boundary to keep only the shared part, or set-difference it away to
    discard the mesh boundary and keep only the interior rim.

    Args:
        mesh: Triangle surface mesh with `anatomical_tags` cell data.
        feature_tag: Value in `anatomical_tags` identifying the region.
        coincides_with_geometry_boundary: Whether to keep the part of the region's boundary lying
            on the mesh's own boundary (`True`) or only the part in the mesh interior (`False`).
        logger: Receives debug messages. Defaults to discarding them.

    Returns:
        Vertex indices into `mesh` of the selected boundary vertices, ordered along the loop,
        shape `(n,)`.
    """
    logger = logging.resolve_logger(logger)
    feature_mesh = mesh.extract_values(feature_tag, scalars="anatomical_tags")
    feature_boundaries = feature_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    geometry_boundaries = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    feature_boundary_inds = feature_boundaries.point_data["vtkOriginalPointIds"]
    geometry_boundary_inds = geometry_boundaries.point_data["vtkOriginalPointIds"]

    logger.debug(
        f"  tag {feature_tag}: {feature_mesh.n_cells} cells, {feature_boundary_inds.size} "
        f"boundary vertices, {geometry_boundary_inds.size} mesh-boundary vertices"
    )
    if coincides_with_geometry_boundary:
        feature_boundary_inds = np.intersect1d(feature_boundary_inds, geometry_boundary_inds)
    else:
        feature_boundary_inds = np.setdiff1d(feature_boundary_inds, geometry_boundary_inds)

    ordered_boundary_inds = _construct_ordered_path_from_indices(
        mesh, feature_boundary_inds, logger
    )
    return ordered_boundary_inds


# --------------------------------------------------------------------------------------------------
def _construct_ordered_path_from_indices(
    mesh: pv.PolyData, path_indices: np.ndarray, logger: logging.BaseLogger
) -> np.ndarray:
    """Reorder an unordered set of boundary-loop vertex indices into travel order along the loop.

    `path_indices` is a set, not a sequence -- pyvista's boundary extraction gives no ordering.
    Restricting the mesh's edges to those with both endpoints in `path_indices` leaves exactly the
    loop's own edges, so a DFS from an arbitrary start vertex over that edge-induced subgraph
    visits every vertex exactly once, in loop order (a boundary loop is a simple cycle, so DFS from
    any vertex traces it without backtracking). If the selection is not one single loop, the DFS
    silently covers only the component containing the first vertex, so the component count is
    logged.
    """
    tm_mesh = tm.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    path_edges = tm_mesh.edges[np.isin(tm_mesh.edges, path_indices).all(axis=1)].flatten()
    local_edges = np.array([np.where(path_indices == ind)[0][0] for ind in path_edges])
    local_edges = local_edges.reshape(-1, 2)
    graph = ig.Graph(local_edges, directed=False)
    logger.debug(
        f"  {path_indices.size} selected vertices form {len(graph.connected_components())} "
        "connected component(s)"
    )
    ordered_path, _ = graph.dfs(0)
    return path_indices[ordered_path]


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData,
    subset_one: np.ndarray,
    subset_two: np.ndarray,
    inadmissible_contact_set: np.ndarray,
    inadmissible_along_set: np.ndarray,
    logger: logging.BaseLogger | None = None,
) -> np.ndarray:
    """Find the shortest edge-path between any vertex of `subset_one` and any of `subset_two`.

    Edges are weighted by their own 3D length, so "shortest" means shortest geodesic-on-the-mesh
    approximation, not fewest hops. `inadmissible_contact_set` removes every edge touching any of
    its vertices at either end (the path may not come near these vertices at all); `inadmissible_
    along_set` removes only edges with *both* endpoints in the set (the path may still touch these
    vertices, e.g. its own start/end boundary, just not travel along that boundary itself). Two
    shortest-path searches are run: an all-pairs distance query to find which `(subset_one,
    subset_two)` vertex pair is actually closest, then a second query for that pair's own path
    (`igraph.get_shortest_paths` doesn't return the path for an arbitrary source/target subset).

    Args:
        mesh: Triangle surface mesh.
        subset_one: Candidate start vertex indices, shape `(n,)`.
        subset_two: Candidate end vertex indices, shape `(m,)`.
        inadmissible_contact_set: Vertex indices the path may not touch. May be empty.
        inadmissible_along_set: Vertex indices the path may touch but not travel along. May be
            empty.
        logger: Receives debug messages. Defaults to discarding them.

    Returns:
        Vertex indices of the path from a `subset_one` vertex to a `subset_two` vertex, shape
        `(k,)`.
    """
    logger = logging.resolve_logger(logger)
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    edges = tm_mesh.edges_unique
    edge_lengths = tm_mesh.edges_unique_length
    inadmissible_contact_edges = np.where(np.isin(edges, inadmissible_contact_set).any(axis=1))[0]
    inadmissible_along_edges = np.where(np.isin(edges, inadmissible_along_set).all(axis=1))[0]
    inadmissible_edges = np.unique(
        np.concatenate([inadmissible_contact_edges, inadmissible_along_edges])
    )
    admissible_edges = np.delete(edges, inadmissible_edges, axis=0)
    admissible_edges_lengths = np.delete(edge_lengths, inadmissible_edges, axis=0)

    graph = ig.Graph(admissible_edges, directed=False)
    distances = np.array(graph.distances(subset_one, subset_two, weights=admissible_edges_lengths))
    rel_start_point, rel_end_point = np.unravel_index(np.argmin(distances), distances.shape)
    logger.debug(
        f"  {inadmissible_edges.size} of {edges.shape[0]} edges removed; closest pair "
        f"{subset_one[rel_start_point]} -> {subset_two[rel_end_point]}, "
        f"distance {distances[rel_start_point, rel_end_point]:.4g}"
    )
    graph = ig.Graph(admissible_edges, directed=False)
    shortest_path = np.array(
        graph.get_shortest_paths(
            subset_one[rel_start_point],
            to=subset_two[rel_end_point],
            weights=admissible_edges_lengths,
        )[0]
    )
    return np.array(shortest_path, dtype=int)


# --------------------------------------------------------------------------------------------------
def extract_region_from_boundary(
    mesh: pv.PolyData,
    boundary_inds: np.ndarray,
    outside_inds: np.ndarray,
    logger: logging.BaseLogger | None = None,
) -> Submesh:
    """Extract the cells enclosed by a boundary loop, on the side away from `outside_inds`.

    Cutting the boundary vertices out of the mesh splits it into (at least) two disconnected
    bodies -- one inside the loop, the rest outside; `split_bodies` finds them, and whichever body
    contains a vertex from `outside_inds` is discarded. A single seed cell
    from the remaining (inside) body then flood-fills outward (`_extract_region_from_boundary`,
    numba-jitted since it walks per-cell face adjacency) up to, but not across, `boundary_inds`'s
    own edges -- needed because the split alone gives the inside body's *vertices*, not the set of
    original-mesh *cells* touching the boundary loop, which the flood-fill recovers by stopping
    exactly at those edges instead of at the (already-removed) boundary vertices.

    Args:
        mesh: Triangle surface mesh.
        boundary_inds: Ordered vertex indices of the closed boundary loop, shape `(n,)`.
        outside_inds: Vertex indices known to lie outside the loop. Only the first is used.
        logger: Receives debug messages. Defaults to discarding them.

    Returns:
        The enclosed region as a `Submesh`; the boundary loop's own cells are included.
    """
    logger = logging.resolve_logger(logger)
    mesh_points_without_boundary = np.setdiff1d(np.arange(mesh.number_of_points), boundary_inds)
    mesh_without_boundary = mesh.extract_points(mesh_points_without_boundary, adjacent_cells=False)
    # `vtkOriginalPointIds` is a VTK-reserved name that `connectivity`/`split_bodies` overwrite
    # rather than pass through; copy it under a plain name first so it survives the split.
    mesh_without_boundary.point_data["original_point_ids"] = mesh_without_boundary.point_data[
        "vtkOriginalPointIds"
    ]
    submeshes = mesh_without_boundary.split_bodies()
    coinciding_outside_inds = np.where(
        submeshes[0].point_data["original_point_ids"] == outside_inds[0]
    )[0]
    inside_mesh = submeshes[0] if coinciding_outside_inds.size == 0 else submeshes[1]
    logger.debug(
        f"  boundary cut splits the mesh into {len(submeshes)} bodies; inside body has "
        f"{inside_mesh.n_points} vertices"
    )
    seed_point = inside_mesh.point_data["original_point_ids"][0]
    seed_ind = np.where(np.isin(mesh.faces.reshape(-1, 4)[:, 1:4], seed_point).any(axis=1))[0][0]

    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4])
    face_adjacency = tm_mesh.face_adjacency
    face_adjacency_flipped = np.flip(face_adjacency, axis=1)
    face_adjacency_complete = np.vstack([face_adjacency, face_adjacency_flipped])
    sorting_mask = np.argsort(face_adjacency_complete[:, 0])
    face_adjacency_sorted = face_adjacency_complete[sorting_mask]
    _, new_face_start_inds = np.unique(face_adjacency_sorted[:, 0], return_index=True)
    neighbor_faces = face_adjacency_sorted[:, 1]
    boundary_edges = np.hstack([boundary_inds[:-1, None], boundary_inds[1:, None]])
    boundary_edges = np.append(boundary_edges, [[boundary_inds[-1], boundary_inds[0]]], axis=0)

    submesh_cell_inds = _extract_region_from_boundary(
        tm_mesh.faces,
        seed_ind,
        boundary_edges,
        new_face_start_inds,
        neighbor_faces,
    )
    pv_submesh = mesh.extract_cells(submesh_cell_inds)
    vertex_inds = pv_submesh.point_data["vtkOriginalPointIds"]
    connectivity = np.array(pv_submesh.cells.reshape(-1, 4)[:, 1:])
    submesh = Submesh(
        vertex_inds=vertex_inds, connectivity=connectivity, cell_inds=submesh_cell_inds
    )

    return submesh


# --------------------------------------------------------------------------------------------------
@njit
def _extract_region_from_boundary(
    faces: np.ndarray,
    seed_ind: int,
    boundary_edges: np.ndarray,
    new_face_start_inds: np.ndarray,
    neighbor_faces: np.ndarray,
) -> np.ndarray:
    """Flood-fill face-adjacency from `seed_ind`, without crossing any edge in `boundary_edges`.

    `neighbor_faces`, grouped by `new_face_start_inds`,
    gives each face's adjacent faces; the shared edge between a cell and a candidate neighbor is
    checked against `boundary_edges` before crossing it, so the fill stops exactly at the boundary.
    Numba-jitted (per-cell adjacency walk over the whole submesh, not vectorizable)
    -- this runs once per segment per pipeline run, not in a hot loop, so plain BFS/DFS is preferred
    over a fancier algorithm.
    """
    visited_cells = np.zeros(new_face_start_inds.shape[0], dtype=np.bool_)
    active_list = List([seed_ind])

    while active_list:
        current_cell_ind = active_list.pop()
        current_cell = faces[current_cell_ind]
        visited_cells[current_cell_ind] = True
        start_ind = new_face_start_inds[current_cell_ind]
        end_ind = (
            new_face_start_inds[current_cell_ind + 1]
            if current_cell_ind + 1 < new_face_start_inds.size
            else neighbor_faces.size
        )
        for i in range(start_ind, end_ind):
            neighbor_cell_ind = neighbor_faces[i]
            neighbor_cell = faces[neighbor_cell_ind]
            shared_edge = np.intersect1d(current_cell, neighbor_cell)
            for boundary_edge in boundary_edges:
                if shared_edge[0] in boundary_edge and shared_edge[1] in boundary_edge:
                    is_boundary = True
                    break
            else:
                is_boundary = False
            if not is_boundary and not visited_cells[neighbor_cell_ind]:
                active_list.append(neighbor_cell_ind)

    return np.where(visited_cells)[0]


# ==================================================================================================
def parameterize_path(
    mesh: pv.PolyData, path: np.ndarray, marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    """Arclength-parameterize `path` so each marker lands at its prescribed relative length.

    The raw path is reordered or reversed first if needed to match the order of `marker_values`.

    Args:
        mesh: Triangle surface mesh providing vertex positions.
        path: Vertex indices of the path, shape `(n,)`, in arbitrary start and direction.
        marker_inds: Vertex indices of the markers, all on `path`.
        marker_values: Relative length in `[0, 1]` for each marker, same order as `marker_inds`.

    Returns:
        The reordered path with a relative length in `[0, 1]` at each vertex.
    """
    reordered_path, relative_marker_inds = _reorder_path_by_markers(
        path, marker_inds, marker_values
    )
    parameterized_path = _parameterize_by_relative_length(
        mesh, reordered_path, relative_marker_inds, marker_values
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def _reorder_path_by_markers(
    path: np.ndarray,
    marker_inds: list[int],
    marker_values: list[float],
) -> tuple[np.ndarray, list[int]]:
    """Roll/flip `path` so it starts at `marker_inds[0]` and travels in `marker_values`' own order.

    `path` (from `get_feature_boundary`/`construct_shortest_path_between_subsets`) has an arbitrary
    start point and travel direction; the markers' *intended* relative-length order is given by
    `marker_values`, so if the raw path visits `marker_inds` in a different relative order than
    `marker_values` says they should be in, the path is travelling the wrong way and gets reversed
    (a boundary loop only has two possible directions once its start point is fixed).
    """
    relative_start_ind_location = np.where(path == marker_inds[0])[0]
    ordered_path = np.roll(path, -relative_start_ind_location)

    relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        ordered_path = np.roll(np.flip(ordered_path), 1)
        relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    return ordered_path, relative_marker_inds


# --------------------------------------------------------------------------------------------------
def _parameterize_by_relative_length(
    mesh: pv.PolyData, path: np.ndarray, relative_marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    """Assign each path vertex a relative length in `[0, 1]`, piecewise linear in arclength.

    Relative length is linear in 3D arclength between consecutive markers (so unevenly spaced mesh
    vertices between two markers still get evenly scaled, not evenly spaced, relative lengths).

    If no marker sits at relative length 1.0, `path` is an open loop that must still close (a
    boundary loop's last vertex connects back to its first) -- the first vertex is appended as a
    synthetic closing marker at `1.0` so the final segment's arclength scaling has an end value to
    interpolate to.
    """
    coordinates = np.array(mesh.points[path])
    if 1.0 not in marker_values:
        path = np.append(path, path[0])
        relative_marker_inds = [*relative_marker_inds, path.size - 1]
        marker_values = [*marker_values, 1.0]
        coordinates = np.append(coordinates, [coordinates[0]], axis=0)
    edge_lengths = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    cumulative_lengths = np.cumsum(edge_lengths)
    cumulative_lengths = np.insert(cumulative_lengths, 0, 0.0)
    segmented_points = np.zeros(path.size)

    num_segments = len(relative_marker_inds) - 1
    for i in range(num_segments):
        start_ind = relative_marker_inds[i]
        end_ind = relative_marker_inds[i + 1]
        start_value = marker_values[i]
        end_value = marker_values[i + 1]
        shifted_cumulative_lengths = (
            cumulative_lengths[start_ind : end_ind + 1] - cumulative_lengths[start_ind]
        )
        relative_lengths = (
            start_value
            + (end_value - start_value)
            * shifted_cumulative_lengths
            / shifted_cumulative_lengths[-1]
        )
        segmented_points[start_ind : end_ind + 1] = relative_lengths

    parameterized_path = ParameterizedPath(path, segmented_points)
    return parameterized_path


# ==================================================================================================
def compute_uacs_polyline(
    path: ParameterizedPath,
    segment_points: list[float],
    segment_uacs: list[tuple[float, float]],
) -> UACPath:
    """Assign (alpha, beta) along `path` by linearly interpolating between `segment_uacs` waypoints.

    `segment_points`/`segment_uacs` are a polyline in UAC space: relative-length breakpoint `i` maps
    to target `segment_uacs[i]`, and every vertex strictly between two breakpoints gets its UAC
    linearly interpolated by its own already-computed relative arclength (`path.relative_lengths`)
    between them -- so the polyline can bend (a different alpha/beta slope per segment) without the
    underlying path itself needing a vertex exactly at each breakpoint's arclength, since `path`'s
    marker-derived relative lengths already land there exactly by construction. Interior breakpoints
    are deduplicated.

    Args:
        path: Arclength-parameterized path to assign coordinates on.
        segment_points: Increasing relative-length breakpoints of the polyline, each present on
            `path`.
        segment_uacs: Target `(alpha, beta)` at each breakpoint, same length as `segment_points`.

    Returns:
        The path from the first to the last breakpoint with (alpha, beta) at each vertex.
    """
    alpha_values, beta_values, ind_values, relative_lengths = [], [], [], []
    num_segments = len(segment_points) - 1

    for i in range(num_segments):
        start_ind = np.where(np.isclose(path.relative_lengths, segment_points[i]))[0][0]
        end_ind = np.where(np.isclose(path.relative_lengths, segment_points[i + 1]))[0][0]
        segment_inds = path.inds[start_ind : end_ind + 1]
        start_uacs = segment_uacs[i]
        end_uacs = segment_uacs[i + 1]
        path_segment = path.relative_lengths[start_ind : end_ind + 1]
        length_range = np.max(path_segment) - np.min(path_segment)
        scaled_length = (path_segment - np.min(path_segment)) / length_range
        alpha = start_uacs[0] + (end_uacs[0] - start_uacs[0]) * scaled_length
        beta = start_uacs[1] + (end_uacs[1] - start_uacs[1]) * scaled_length

        if i < num_segments - 1:
            alpha = alpha[:-1]
            beta = beta[:-1]
            segment_inds = segment_inds[:-1]
            path_segment = path_segment[:-1]
        alpha_values.append(alpha)
        beta_values.append(beta)
        ind_values.append(segment_inds)
        relative_lengths.append(path_segment)

    ind_values = np.concatenate(ind_values)
    relative_lengths = np.concatenate(relative_lengths)
    alpha_values = np.concatenate(alpha_values)
    beta_values = np.concatenate(beta_values)
    uac_path = UACPath(
        inds=ind_values, relative_lengths=relative_lengths, alpha=alpha_values, beta=beta_values
    )
    return uac_path

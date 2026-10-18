"""Mesh preprocessing: decimation/smoothing plus cleanup of the resulting `anatomical_tags` defects.

Mesh coordinates are patient-space (whatever units the input mesh was authored in) until
`preprocess_mesh`'s own `dimension_scaling_factor` rescales them -- everything downstream of this
module (`construction/`, `basis.py`) works in the rescaled space.
"""

from numbers import Real

import fast_simplification as fs
import numpy as np
import pyvista as pv
import trimesh as tm


# ==================================================================================================
def convert_unstructured_to_polydata_mesh(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    """Convert an `UnstructuredGrid` of triangles into a `PolyData` surface mesh.

    Several pyvista filters used across this codebase (`extract_values`, `split_bodies`, ...)
    return an `UnstructuredGrid`, but the rest of the pipeline is written against `PolyData`'s API
    (`.faces`, `from_regular_faces`) -- this is the one place that converts back.

    Args:
        mesh: Grid whose cells are all triangles.

    Returns:
        A `PolyData` with the same points and cells, and all point and cell data carried over
        unchanged.
    """
    point_data = mesh.point_data
    cell_data = mesh.cell_data
    polydata_mesh = pv.PolyData(mesh.points, mesh.cells)
    for key in point_data:
        polydata_mesh.point_data[key] = point_data[key]
    for key in cell_data:
        polydata_mesh.cell_data[key] = cell_data[key]

    return polydata_mesh


# --------------------------------------------------------------------------------------------------
def preprocess_mesh(
    mesh: pv.PolyData,
    tag_values: list[int],
    decimation_factor: Real,
    num_smoothing_iters: int = 50,
    extraction_threshold: Real = 0.5,
    dimension_scaling_factor: Real = 1e-3,
) -> pv.PolyData:
    """Decimate/smooth `mesh`, carrying its `anatomical_tags`/`fibers` cell data onto the result.

    Decimation and smoothing (`fast_simplification`/pyvista's Laplacian smoothing) don't preserve
    cell data on their own, so each tag's region is instead re-derived on the coarse mesh:
    interpolate the fine mesh's tag indicator (as point data, since `interpolate` works
    point-wise) onto the coarse mesh, then threshold at `extraction_threshold` of the tag's own
    value to decide which coarse cells now belong to it. Doing this per tag (rather than
    interpolating all tags simultaneously) avoids one tag's region bleeding into another's at a
    shared boundary. Fiber vectors are carried the same way, then re-projected onto each coarse
    cell's own tangent plane (`_project_to_tangent_space`) since interpolation alone doesn't keep
    them exactly tangent to the now-different local geometry. `fix_feature_tag_defects` cleans up
    the small tag-boundary artifacts (dangling fragments, single-cell spikes) this
    interpolate-and-threshold approach leaves behind. `mesh.points` are in patient-space
    coordinates until `dimension_scaling_factor` rescales them here -- downstream code assumes the
    rescaled units.

    Args:
        mesh: Fine surface mesh in patient-space units with `anatomical_tags` (integer, one per
            cell) and `fibers` (shape `(n_cells, 3)`) cell data.
        tag_values: Tag values present in `anatomical_tags`. The value `0` (untagged body) is
            skipped.
        decimation_factor: Fraction of cells to remove in decimation, in `[0, 1)`
            (`fast_simplification`'s `target_reduction`).
        num_smoothing_iters: Number of Laplacian smoothing iterations on the decimated mesh.
        extraction_threshold: Fraction of a tag's value above which an interpolated cell counts
            as belonging to that tag, in `(0, 1]`.
        dimension_scaling_factor: Factor multiplied onto the point coordinates, e.g. `1e-3` to
            convert millimeters to meters.

    Returns:
        The coarse, smoothed, rescaled mesh with `anatomical_tags` (integer per cell) and unit
        tangent `fibers` (shape `(n_cells, 3)`) cell data.
    """
    tag_data = mesh.cell_data["anatomical_tags"]
    fine_mesh = mesh.copy()
    fine_mesh.cell_data["anatomical_tags"] = 0
    coarse_mesh = fs.simplify_mesh(mesh, target_reduction=decimation_factor)
    smoothened_mesh = coarse_mesh.smooth(n_iter=num_smoothing_iters, relaxation_factor=0.1)
    smoothened_mesh.cell_data["anatomical_tags"] = np.zeros(smoothened_mesh.number_of_cells)

    for tag_value in tag_values:
        if tag_value == 0:
            continue
        tag_mask = tag_data == tag_value
        fine_mesh.cell_data["anatomical_tags"][tag_mask] = tag_value
        fine_mesh_point_data = fine_mesh.cell_data_to_point_data()
        interpolated_mesh = smoothened_mesh.interpolate(
            fine_mesh_point_data, strategy="closest_point"
        )
        feature_mesh = interpolated_mesh.extract_values(
            ranges=[extraction_threshold * tag_value, float("inf")], scalars="anatomical_tags"
        )
        smoothened_mesh.cell_data["anatomical_tags"][
            feature_mesh.cell_data["vtkOriginalCellIds"]
        ] = tag_value
        fine_mesh.cell_data["anatomical_tags"][tag_mask] = 0

    interpolated_mesh = interpolated_mesh.point_data_to_cell_data()
    fiber_vectors = _project_to_tangent_space(
        interpolated_mesh.cell_data["fibers"], smoothened_mesh
    )
    fiber_vectors = fiber_vectors / np.linalg.norm(fiber_vectors, axis=1)[:, None]
    smoothened_mesh.cell_data["fibers"] = fiber_vectors
    smoothened_mesh.points *= dimension_scaling_factor

    smoothened_mesh = fix_feature_tag_defects(smoothened_mesh)
    return smoothened_mesh


# --------------------------------------------------------------------------------------------------
def fix_feature_tag_defects(mesh: pv.PolyData) -> pv.PolyData:
    """Clean up the small tag-boundary artifacts left by `preprocess_mesh`'s tag transfer.

    Each stage fixes a defect the next stage's own detection would otherwise misread as a
    legitimate feature.

    Order matters: dangling fragments (`_fix_feature_tag_dangles`) are removed first because a
    disconnected fleck of a tag elsewhere on the mesh would otherwise look like a valid extra
    "region" rather than noise; spikes on that now-cleaned tag boundary (`_fix_feature_tag_spikes`)
    and on the untagged body's own boundary (`_fix_body_tag_spikes`) are single-cell protrusions
    that would otherwise distort a feature's extracted boundary loop (`geometry.get_feature_
    boundary`); and mesh-wide single-neighbor cells (`_fix_boundary_spikes`), independent of any
    tag, are removed last since they're a general mesh-quality defect this pipeline's boundary/path
    extraction (`geometry.py`) can't handle regardless of tagging.

    Args:
        mesh: Surface mesh with `anatomical_tags` cell data. Modified in place by the tag-fixing
            stages.

    Returns:
        The cleaned mesh. Cells (and the points only they used) may be removed by the last stage,
        so cell indices are not preserved.
    """
    mesh = _fix_feature_tag_dangles(mesh)
    mesh = _fix_feature_tag_spikes(mesh)
    mesh = _fix_body_tag_spikes(mesh)
    mesh = _fix_boundary_spikes(mesh)
    return mesh


# ==================================================================================================
def _fix_boundary_spikes(mesh: pv.PolyData) -> pv.PolyData:
    """Drop every cell with at most one neighbor -- a single-cell spike jutting off the mesh body.

    Unlike `_fix_feature_tag_spikes`/`_fix_body_tag_spikes`, this isn't about tag assignment at
    all: a spike cell (however tagged) breaks the one-boundary-loop-per-feature assumption
    `geometry._construct_ordered_path_from_indices`'s DFS relies on, so it's removed from the mesh
    outright rather than reassigned to a different tag.

    Uses `extract_cells` rather than reassigning `mesh.faces` directly: the latter changes cell
    *topology* without touching `cell_data` at all, silently desyncing e.g. `anatomical_tags`/
    `fibers` from the cells they actually belong to for every cell after the first removed spike;
    `extract_cells` subsets cell/point data consistently along with the topology.
    """
    num_neighbors_per_cell = _get_num_neighbors_per_cell(mesh)
    non_spike_cells = np.where(num_neighbors_per_cell > 1)[0]
    mesh = convert_unstructured_to_polydata_mesh(mesh.extract_cells(non_spike_cells))
    mesh = mesh.remove_unused_points()

    return mesh


# --------------------------------------------------------------------------------------------------
def _fix_feature_tag_dangles(mesh: pv.PolyData) -> pv.PolyData:
    """Keep only each tag's largest connected component, discarding smaller disconnected pieces.

    The interpolate-and-threshold reassignment in `preprocess_mesh` can leave a handful of cells
    tagged the same as a feature despite sitting elsewhere on the mesh, disconnected from that
    feature's main body -- `trimesh.split` finds these components, and cells outside the largest one
    are untagged (their tag is left at `0` in `new_mesh`, filled in only for the largest component).
    `fill_holes` closes any gaps left in the main body's own triangulation by cells that got
    reassigned away from it, so its boundary loop stays simple before `find_containing_cell` maps
    the (possibly hole-filled) component back onto the original mesh's own cell indices.
    """
    feature_tags = np.unique(mesh.cell_data["anatomical_tags"])
    feature_tags = feature_tags[feature_tags > 0]
    new_mesh = mesh.copy()
    new_mesh.cell_data["anatomical_tags"] = np.zeros(mesh.number_of_cells, dtype=int)

    for tag_value in feature_tags:
        feature_mesh = mesh.extract_values(tag_value, scalars="anatomical_tags")
        tm_mesh = tm.Trimesh(
            vertices=feature_mesh.points, faces=feature_mesh.cells.reshape(-1, 4)[:, 1:]
        )
        split_mesh = tm_mesh.split(only_watertight=False)
        num_cells = [submesh.faces.shape[0] for submesh in split_mesh]
        main_body = split_mesh[np.argmax(num_cells)]
        main_body.fill_holes()
        original_cell_inds = new_mesh.find_containing_cell(main_body.triangles_center)
        new_mesh.cell_data["anatomical_tags"][original_cell_inds] = tag_value

    return new_mesh


# --------------------------------------------------------------------------------------------------
def _fix_feature_tag_spikes(mesh: pv.PolyData) -> pv.PolyData:
    """Untag any feature cell touching its own tagged region at only one edge (a boundary spike).

    A spike cell dangling off a feature's boundary would otherwise appear in that feature's
    extracted boundary loop as a degenerate one-cell-wide detour; untagging it (reassigning to the
    untagged body, tag `0`) removes it from the region's own submesh entirely. Repeats until a full
    pass fixes nothing, since untagging one spike can expose a former interior cell as the new
    single-neighbor spike.
    """
    feature_tags = np.unique(mesh.cell_data["anatomical_tags"])
    feature_tags = feature_tags[feature_tags > 0]
    num_fixed_cells = 1

    while num_fixed_cells > 0:
        num_fixed_cells = 0
        for tag_value in feature_tags:
            feature_mesh = mesh.extract_values(tag_value, scalars="anatomical_tags")
            feature_mesh = convert_unstructured_to_polydata_mesh(feature_mesh)
            num_neighbors_per_cell = _get_num_neighbors_per_cell(feature_mesh)
            spike_cells = np.where(num_neighbors_per_cell == 1)[0]
            mesh.cell_data["anatomical_tags"][
                feature_mesh.cell_data["vtkOriginalCellIds"][spike_cells]
            ] = 0
            num_fixed_cells += spike_cells.shape[0]

    return mesh


# --------------------------------------------------------------------------------------------------
def _fix_body_tag_spikes(mesh: pv.PolyData) -> pv.PolyData:
    """Reassign single-neighbor spikes in the untagged body (tag `0`) to their surrounding tag.

    The untagged-body mirror of `_fix_feature_tag_spikes`.

    An untagged spike poking into a feature is exactly as disruptive to that feature's boundary
    loop as a feature spike poking out (`_fix_feature_tag_spikes`), just from the other side, so it
    is filled in with `_get_different_neighbor_tag`'s tag rather than removed. Repeats to a fixed
    point for the same reason as `_fix_feature_tag_spikes`.
    """
    num_fixed_cells = 1

    while num_fixed_cells > 0:
        num_fixed_cells = 0
        body_mesh = mesh.extract_values(0, scalars="anatomical_tags")
        body_mesh = convert_unstructured_to_polydata_mesh(body_mesh)
        num_neighbors_per_cell = _get_num_neighbors_per_cell(body_mesh)
        spike_cells = np.where(num_neighbors_per_cell == 1)[0]
        original_spike_cells = body_mesh.cell_data["vtkOriginalCellIds"][spike_cells]
        new_tags = _get_different_neighbor_tag(mesh, original_spike_cells)
        mesh.cell_data["anatomical_tags"][original_spike_cells] = new_tags
        num_fixed_cells += spike_cells.shape[0]

    return mesh


# --------------------------------------------------------------------------------------------------
def _get_different_neighbor_tag(mesh: pv.PolyData, cell_inds: np.ndarray) -> np.ndarray:
    """Find, for each given cell, a neighboring cell's tag that differs from its own.

    Assumes each `cell_inds` entry is a single-neighbor-tag spike (guaranteed by
    `_fix_body_tag_spikes`'s caller): it has exactly one surrounding tag other than its own, so
    the first differing neighbor tag found is the correct reassignment, not an arbitrary pick.
    """
    different_tags = []
    for ind in cell_inds:
        cell_tag = mesh.cell_data["anatomical_tags"][ind]
        cell_neighbors = mesh.cell_neighbors(ind)
        cell_neighbor_tags = mesh.cell_data["anatomical_tags"][cell_neighbors]
        different_tag = cell_neighbor_tags[cell_neighbor_tags != cell_tag][0]
        different_tags.append(different_tag)

    return np.array(different_tags)


# --------------------------------------------------------------------------------------------------
def _get_num_neighbors_per_cell(mesh: pv.PolyData) -> np.ndarray:
    """Count the face-adjacent neighbors of every cell."""
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    face_adjacency_data = tm_mesh.face_adjacency
    num_neighbors_per_cell = np.bincount(
        face_adjacency_data.flatten(), minlength=mesh.number_of_cells
    )
    return num_neighbors_per_cell


# --------------------------------------------------------------------------------------------------
def _project_to_tangent_space(vectors: np.ndarray, mesh: pv.PolyData) -> np.ndarray:
    """Remove each cell's normal component from its vector, leaving the tangent part.

    Interpolating a fiber vector onto a coarser, smoothed mesh doesn't keep it exactly tangent to
    the new (slightly different) local geometry, so this re-tangentializes it afterward -- the
    interpolated vector's magnitude changes slightly (a normal component is discarded) but its
    tangent direction, the physically meaningful part of a fiber orientation, is preserved.
    """
    cell_normals = mesh.cell_normals
    projected_vectors = (
        vectors - np.einsum("ij,ij->i", vectors, cell_normals)[:, None] * cell_normals
    )
    return projected_vectors

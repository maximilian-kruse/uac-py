"""Unit tests for `construction/geometry.py`, on small synthetic meshes.

No real patient mesh needed: the invariants under test (arclength parameterization, polyline
interpolation, boundary/region extraction) hold for any valid triangulation, so a planar grid or a
bare polyline stands in for the anatomy.
"""

import numpy as np
import pyvista as pv
from numpy.testing import assert_allclose

from uacpy.construction import geometry, model


def _unit_square_loop() -> pv.PolyData:
    """Four points on the corners of the unit square; only `points` is read by the path code."""
    return pv.PolyData(np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))


def _tagged_grid(n: int, tagged_squares: set[tuple[int, int]]) -> pv.PolyData:
    """An `n x n`-vertex planar grid of unit squares, two triangles each, with `anatomical_tags`
    1 on the triangles of `tagged_squares` (given as `(row, column)`) and 0 elsewhere.

    Each square is cut along the diagonal through its corner nearest the grid center, so for a
    tagged block centered on the grid no mesh edge cuts across a corner of the block's rim (the
    rim then really is a simple cycle in the mesh graph).
    """
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    center = np.array([(n - 1) / 2, (n - 1) / 2, 0.0])
    faces, tags = [], []
    for row in range(n - 1):
        for column in range(n - 1):
            lower_left = row * n + column
            # counter-clockwise corners; the diagonal joins corner k and its opposite k + 2
            corners = [lower_left, lower_left + 1, lower_left + n + 1, lower_left + n]
            nearest = int(np.argmin([np.linalg.norm(points[c] - center) for c in corners])) % 2
            a, b, c, d = (corners[(nearest + k) % 4] for k in range(4))
            faces += [[a, b, c], [a, c, d]]
            tags += [int((row, column) in tagged_squares)] * 2
    mesh = pv.PolyData.from_regular_faces(points, np.array(faces))
    mesh.cell_data["anatomical_tags"] = np.array(tags)
    # `get_feature_boundary` reads this array off the *whole* mesh's boundary edges, which pyvista
    # only fills in if the mesh already carries it (the patient meshes do, from the filter chain
    # that wrote them); a freshly built mesh has to supply it itself.
    mesh.point_data["vtkOriginalPointIds"] = np.arange(mesh.n_points)
    return mesh


GRID_SIZE = 8
BLOCK_SQUARES = {(row, column) for row in range(2, 5) for column in range(2, 5)}
BLOCK_RIM = {row * GRID_SIZE + column for row in range(2, 6) for column in range(2, 6)} - {
    row * GRID_SIZE + column for row in range(3, 5) for column in range(3, 5)
}


def test_parameterize_path_is_linear_in_arclength_and_closes_the_loop():
    """With one marker at 0 and one at 0.5, the open loop is closed back to its first vertex at
    relative length 1, and the equal-length sides of the square get equal relative lengths."""
    path = geometry.parameterize_path(
        _unit_square_loop(), np.array([0, 1, 2, 3]), marker_inds=[0, 2], marker_values=[0.0, 0.5]
    )

    assert_allclose(path.inds, [0, 1, 2, 3, 0])
    assert_allclose(path.relative_lengths, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_parameterize_path_reverses_a_path_that_runs_against_the_marker_order():
    """The raw path visits marker 3 before marker 2, but the prescribed values say marker 2 comes
    first, so the path has to be traversed in the other direction."""
    path = geometry.parameterize_path(
        _unit_square_loop(),
        np.array([0, 1, 2, 3]),
        marker_inds=[0, 3, 2],
        marker_values=[0.0, 0.25, 0.5],
    )

    assert_allclose(path.inds, [0, 3, 2, 1, 0])
    assert_allclose(path.relative_lengths, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_compute_uacs_polyline_interpolates_linearly_between_waypoints_and_bends_at_them():
    """The polyline (0,0) -> (1,0) -> (1,1) changes slope at the middle waypoint, which appears
    only once in the output even though both neighbouring segments contain it."""
    path = model.ParameterizedPath(
        inds=np.array([0, 1, 2, 3, 0]), relative_lengths=np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    )

    uac_path = geometry.compute_uacs_polyline(
        path, segment_points=[0.0, 0.5, 1.0], segment_uacs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    )

    assert_allclose(uac_path.inds, [0, 1, 2, 3, 0])
    assert_allclose(uac_path.alpha, [0.0, 0.5, 1.0, 1.0, 1.0])
    assert_allclose(uac_path.beta, [0.0, 0.0, 0.0, 0.5, 1.0])


def test_get_feature_boundary_returns_the_ordered_rim_of_an_interior_region():
    """A 3x3 block of squares in the middle of an 8x8-vertex grid is tagged: its rim is the 12
    outer vertices of the block's 4x4 vertices, visited as a closed loop (consecutive vertices,
    including last to first, are one grid step apart)."""
    mesh = _tagged_grid(GRID_SIZE, BLOCK_SQUARES)

    boundary = geometry.get_feature_boundary(mesh, 1, coincides_with_geometry_boundary=False)

    assert set(boundary) == BLOCK_RIM
    closed_loop = np.append(boundary, boundary[0])
    steps = np.linalg.norm(np.diff(mesh.points[closed_loop], axis=0), axis=1)
    assert_allclose(steps, 1.0)


def test_extract_region_from_boundary_recovers_exactly_the_cells_inside_the_loop():
    """Cutting along the rim of the tagged block and flood-filling from the inside side returns
    the tagged cells (rim cells included) and nothing outside them."""
    mesh = _tagged_grid(GRID_SIZE, BLOCK_SQUARES)
    boundary = geometry.get_feature_boundary(mesh, 1, coincides_with_geometry_boundary=False)

    submesh = geometry.extract_region_from_boundary(mesh, boundary, outside_inds=np.array([0]))

    tagged_cells = np.where(mesh.cell_data["anatomical_tags"] == 1)[0]
    assert_allclose(np.sort(submesh.cell_inds), tagged_cells)
    assert set(submesh.vertex_inds) == {
        row * GRID_SIZE + column for row in range(2, 6) for column in range(2, 6)
    }

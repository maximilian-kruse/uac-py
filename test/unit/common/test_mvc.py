"""Unit tests for `mvc.compute_mean_value_coordinates`, the mean-value-coordinate parameterization.

No mesh fixture needed: both tests build a small synthetic planar grid mesh directly, since the
invariants under test (boundary reproduction, affine/linear precision) hold for any triangulation.
"""

import numpy as np
from numpy.testing import assert_allclose

from uacpy.common import mvc


def _build_grid_mesh(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An n x n grid of jittered points, triangulated into two triangles per cell.

    Returns `(vertices, simplices, boundary_inds)`: `vertices` are `(n * n, 2)` planar
    coordinates, `simplices` index them into triangles, and `boundary_inds` are the indices
    lying on the grid's outer perimeter. Interior points are perturbed off the regular grid (with
    a fixed seed) so the mesh has no special symmetry that could mask an implementation bug that
    a perfectly regular grid's angles happen to cancel out.
    """
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    vertices = np.column_stack((x.ravel(), y.ravel())).astype(float)
    is_interior = ((x > 0) & (x < n - 1) & (y > 0) & (y < n - 1)).ravel()
    jitter = np.random.default_rng(1).uniform(-0.3, 0.3, size=(vertices.shape[0], 2))
    vertices[is_interior] += jitter[is_interior]

    def vertex_index(i: int, j: int) -> int:
        """Flat vertex index of grid node `(i, j)`."""
        return i * n + j

    simplices = []
    for i in range(n - 1):
        for j in range(n - 1):
            bottom_left = vertex_index(i, j)
            bottom_right = vertex_index(i, j + 1)
            top_left = vertex_index(i + 1, j)
            top_right = vertex_index(i + 1, j + 1)
            simplices.append((bottom_left, bottom_right, top_right))
            simplices.append((bottom_left, top_right, top_left))
    simplices = np.array(simplices)

    is_boundary = (x == 0) | (x == n - 1) | (y == 0) | (y == n - 1)
    boundary_inds = np.flatnonzero(is_boundary.ravel())
    return vertices, simplices, boundary_inds


def test_mvc_reproduces_boundary_values_exactly():
    """The result at `boundary_inds` must equal `boundary_coordinates`, for arbitrary targets."""
    vertices, simplices, boundary_inds = _build_grid_mesh(5)
    rng = np.random.default_rng(0)
    boundary_coordinates = rng.uniform(-1, 1, size=(boundary_inds.shape[0], 2))

    uacs = mvc.compute_mean_value_coordinates(
        vertices, simplices, boundary_inds, boundary_coordinates
    )

    assert_allclose(uacs[boundary_inds], boundary_coordinates)


def test_mvc_reproduces_affine_functions_exactly():
    """Mean value coordinates are generalized barycentric coordinates: for any vertex, the
    (normalized) weighted sum of its one-ring neighbours' own positions reproduces that vertex's
    own position exactly. Solving the harmonic system with the mesh's own planar coordinates as
    the boundary condition must therefore recover those same coordinates everywhere, not only on
    the boundary -- a much stronger check than boundary reproduction alone, since it exercises the
    interior linear solve against a solution that's known in closed form.
    """
    vertices, simplices, boundary_inds = _build_grid_mesh(6)
    boundary_coordinates = vertices[boundary_inds]

    uacs = mvc.compute_mean_value_coordinates(
        vertices, simplices, boundary_inds, boundary_coordinates
    )

    assert_allclose(uacs, vertices, atol=1e-9)

"""Unit test for `jacobian.compute_cellwise_jacobians`, the per-triangle affine Jacobian.

No mesh fixture needed: the invariant under test (exactness for affine maps) holds for any
non-degenerate triangulation, so this generates its own random triangles.
"""

import numpy as np
from numpy.testing import assert_allclose

from uacpy.common import jacobian


def test_cellwise_jacobians_are_exact_for_an_affine_map():
    """Recovering the Jacobian of a known affine 2D-to-3D map must return that exact map.

    `compute_cellwise_jacobians` solves, per triangle, for the single 3x2 matrix that maps that
    triangle's own two 2D edges onto its own two 3D edges. If the 3D vertices are actually
    `linear_map @ vertices_2d + offset` for one fixed `linear_map`/`offset` shared by every
    triangle -- an affine embedding of the whole 2D domain into 3D -- then the per-triangle
    solve should recover exactly that shared `linear_map`, regardless of each triangle's own
    shape, size, or orientation.
    """
    rng = np.random.default_rng(0)
    linear_map = rng.uniform(-2, 2, size=(3, 2))
    offset = rng.uniform(-2, 2, size=3)

    num_triangles = 20
    vertices_2d = rng.uniform(-1, 1, size=(num_triangles, 3, 2))
    vertices_3d = vertices_2d @ linear_map.T + offset

    simplices = np.arange(num_triangles * 3).reshape(num_triangles, 3)
    jacobians = jacobian.compute_cellwise_jacobians(
        vertices_3d.reshape(-1, 3), vertices_2d.reshape(-1, 2), simplices
    )

    assert_allclose(jacobians, np.broadcast_to(linear_map, jacobians.shape), atol=1e-9)

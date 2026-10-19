"""Unit test for `model.UACSubmesh.to_polydata`, the flat (alpha, beta) mesh used for checks."""

import numpy as np
from numpy.testing import assert_allclose

from uacpy.construction import model


def test_to_polydata_places_vertices_at_alpha_beta_on_the_z_zero_plane():
    submesh = model.UACSubmesh(
        vertex_inds=np.array([4, 5, 6]),
        cell_inds=np.array([0]),
        connectivity=np.array([[0, 1, 2]]),
        alpha=np.array([0.0, 2.0, 0.0]),
        beta=np.array([0.0, 0.0, 3.0]),
    )

    flat = submesh.to_polydata()

    assert_allclose(flat.points, [[0, 0, 0], [2, 0, 0], [0, 3, 0]])
    assert flat.n_cells == 1
    assert_allclose(flat.area, 3.0)

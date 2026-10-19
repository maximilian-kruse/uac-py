"""Unit tests for the pure-numeric helpers in `basis.py`.

`construct_cellwise_basis` itself needs a real tagged mesh plus the `igl`/`meshlib` hole-filling
pipeline, so it's only exercised indirectly (via `la_basis.ipynb`). These two helpers are pure
math with no mesh I/O, so -- despite being private -- they're worth testing directly here; there's
no cheaper way to reach this logic than through the full mesh pipeline otherwise.
"""

import numpy as np
from numpy.testing import assert_allclose

from uacpy import basis


def test_map_path_to_circle_places_points_evenly_spaced_on_the_circle():
    """Every mapped point lies on the circle, evenly spaced in angle."""
    radius = 2.5
    path = list(range(8))

    x_coordinates, y_coordinates = basis._map_path_to_circle(radius, path)

    assert_allclose(np.hypot(x_coordinates, y_coordinates), radius)
    angles = np.unwrap(np.arctan2(y_coordinates, x_coordinates))
    assert_allclose(np.diff(angles), 2 * np.pi / len(path))


def test_orthonormalize_jacobian_columns_preserves_the_tangent_plane():
    """The output basis must be orthonormal and span the same 2D subspace as the input Jacobian --
    properties any valid orthonormalization must have, independent of the particular algorithm
    (Gram-Schmidt here) used to get there.
    """
    rng = np.random.default_rng(0)
    jacobians = rng.uniform(-2, 2, size=(10, 3, 2))

    basis_vectors = basis._orthonormalize_jacobian_columns(jacobians)

    gram = np.einsum("nik,nil->nkl", basis_vectors, basis_vectors)
    assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-9)

    # Same column space as the input: projecting the input onto the new basis and back must
    # reproduce it exactly, which only holds if the input's columns already lie in that span.
    coefficients = np.einsum("nik,nil->nkl", basis_vectors, jacobians)
    reconstructed = np.einsum("nik,nkl->nil", basis_vectors, coefficients)
    assert_allclose(reconstructed, jacobians, atol=1e-9)

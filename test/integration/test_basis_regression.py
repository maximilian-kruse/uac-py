"""Integration test for `basis.construct_cellwise_basis`, on the same patient mesh
`test_la_construction.py`'s golden-snapshot pipeline test uses.

Unlike `test_basis.py`'s pure-math unit tests of its two private helpers, this exercises the whole
function -- feature-tag boundary detection, `meshlib` hole-filling, and the whole-mesh MVC solve --
which needs a real tagged mesh and can't be reached any cheaper. `data/patient_01/basis_vecs.npy`
is a committed snapshot of this function's own prior output on this same mesh (originally produced
by `notebooks/la_basis.ipynb`): a mismatch here means either a real change to
`construct_cellwise_basis` (investigate) or a deliberate change to the input mesh (regenerate the
snapshot -- never loosen the tolerance to make it pass).
"""

import numpy as np
import pytest
import pyvista as pv
from numpy.testing import assert_allclose

from uacpy import basis
from uacpy.common import mesh_utils

MESH_PATH = "data/patient_01/mesh_with_fibers_tags.vtu"
BASIS_SNAPSHOT_PATH = "data/patient_01/basis_vecs.npy"
MITRAL_VALVE_TAG = 0
EXTERIOR_BOUNDARY_RADIUS = 10


@pytest.fixture(scope="module")
def cellwise_basis():
    """The `(basis_vectors, mvc_coordinates, cellwise_jacobians)` result for the patient mesh."""
    mesh = pv.read(MESH_PATH)
    mesh = mesh_utils.convert_unstructured_to_polydata_mesh(mesh)
    return basis.construct_cellwise_basis(mesh, MITRAL_VALVE_TAG, EXTERIOR_BOUNDARY_RADIUS)


@pytest.mark.integration
@pytest.mark.slow
def test_basis_vectors_are_orthonormal_per_cell(cellwise_basis):
    """The same orthonormality invariant `test_basis.py` checks on synthetic Jacobians, now
    holding for the real per-cell bases this function derives from an actual patient mesh.
    """
    basis_vectors, _, _ = cellwise_basis
    gram = np.einsum("nik,nil->nkl", basis_vectors, basis_vectors)
    assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-9)


@pytest.mark.integration
@pytest.mark.slow
def test_basis_vectors_span_the_same_tangent_plane_as_the_raw_jacobian(cellwise_basis):
    """Projecting the raw Jacobian onto the orthonormalized basis and back must reproduce it
    exactly, for every cell of a real mesh -- the same round-trip invariant `test_basis.py`
    checks on synthetic data.
    """
    basis_vectors, _, cellwise_jacobians = cellwise_basis
    coefficients = np.einsum("nik,nil->nkl", basis_vectors, cellwise_jacobians)
    reconstructed = np.einsum("nik,nkl->nil", basis_vectors, coefficients)
    assert_allclose(reconstructed, cellwise_jacobians, atol=1e-9)


@pytest.mark.integration
@pytest.mark.slow
def test_basis_vectors_match_snapshot(cellwise_basis):
    """The basis vectors equal the pinned snapshot; regenerate it only for a deliberate change."""
    basis_vectors, _, _ = cellwise_basis
    snapshot = np.load(BASIS_SNAPSHOT_PATH)

    assert basis_vectors.shape == snapshot.shape, "mesh changed -- regenerate the snapshot"
    assert_allclose(basis_vectors, snapshot, atol=1e-9)

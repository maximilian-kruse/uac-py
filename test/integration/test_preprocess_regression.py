"""Regression test for `mesh_utils.preprocess_mesh`, as run in `notebooks/preprocess_mesh.ipynb`.

The raw `patient_02` mesh is re-tagged to the pipeline's convention and preprocessed with the
notebook's settings. Pinned point/cell counts and per-tag cell counts catch a change in the
preprocessing (investigate); regenerate them only for a deliberate change to the raw mesh or the
settings -- never loosen the assertions to make them pass.
"""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from numpy.testing import assert_allclose

from uacpy.common import mesh_utils

DATA_DIR = Path("data")
RAW_MESH_PATH = DATA_DIR / "patient_02" / "mesh_endo_raw.vtu"
LEGACY_TAGS = {"MV": 11, "LAA": 13, "LIPV": 21, "LSPV": 23, "RIPV": 25, "RSPV": 27}
NEW_TAGS = {"MV": 0, "LAA": 1, "LIPV": 2, "LSPV": 3, "RIPV": 4, "RSPV": 5}
DIMENSION_SCALING_FACTOR = 1e-3

# Pinned from one run of the settings above.
EXPECTED_POINT_COUNT = 2942
EXPECTED_CELL_COUNT = 5792
EXPECTED_CELLS_PER_TAG = {0: 3666, 1: 1135, 2: 387, 3: 229, 4: 251, 5: 124}


@pytest.fixture(scope="module")
def raw_and_preprocessed_mesh():
    raw = mesh_utils.convert_unstructured_to_polydata_mesh(pv.read(RAW_MESH_PATH))
    del raw.point_data["alpha"]
    del raw.point_data["beta"]
    tags = raw.cell_data["anatomical_tags"]
    for name, legacy_value in LEGACY_TAGS.items():
        tags[tags == legacy_value] = NEW_TAGS[name]
    raw_copy = raw.copy()
    preprocessed = mesh_utils.preprocess_mesh(
        raw,
        tag_values=NEW_TAGS.values(),
        decimation_factor=1,
        num_smoothing_iters=0,
        extraction_threshold=0.75,
        dimension_scaling_factor=DIMENSION_SCALING_FACTOR,
    )
    return raw_copy, preprocessed


@pytest.mark.slow
def test_preprocessed_fibers_are_unit_tangent_vectors(raw_and_preprocessed_mesh):
    _, mesh = raw_and_preprocessed_mesh
    fibers = np.array(mesh.cell_data["fibers"])
    normals = np.array(
        mesh.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"]
    )

    assert_allclose(np.linalg.norm(fibers, axis=1), 1.0, atol=1e-6)
    assert_allclose(np.einsum("ij,ij->i", fibers, normals), 0.0, atol=1e-6)


@pytest.mark.slow
def test_preprocessing_rescales_the_mesh_and_keeps_every_tag(raw_and_preprocessed_mesh):
    raw, mesh = raw_and_preprocessed_mesh

    assert_allclose(
        np.array(mesh.bounds), np.array(raw.bounds) * DIMENSION_SCALING_FACTOR, rtol=0.05
    )
    assert set(np.unique(mesh.cell_data["anatomical_tags"])) == set(NEW_TAGS.values())


@pytest.mark.slow
def test_preprocessed_mesh_matches_snapshot(raw_and_preprocessed_mesh):
    _, mesh = raw_and_preprocessed_mesh
    tags, counts = np.unique(mesh.cell_data["anatomical_tags"], return_counts=True)

    assert mesh.n_points == EXPECTED_POINT_COUNT
    assert mesh.n_cells == EXPECTED_CELL_COUNT
    assert dict(zip(tags.tolist(), counts.tolist(), strict=True)) == EXPECTED_CELLS_PER_TAG

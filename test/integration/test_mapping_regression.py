"""Regression test for cross-patient fiber transfer, as run in `notebooks/la_mapping.ipynb`.

Each source patient's fibers are moved onto `patient_01`'s mesh through the committed UAC
submesh data of both. The invariants (unit length, tangent to the target cell) are checked for all
sources; a pinned snapshot (`snapshots/mapped_fibers_to_patient_01.npz`, float32 to keep it small,
for a subset of sources) additionally catches any change in the values. A mismatch means either a
real change to `mapper.py` or the serialization (investigate), or a deliberate change to the UAC
data (regenerate with `pixi run -e test python test/integration/test_mapping_regression.py` --
never loosen the tolerance to make the test pass).
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from uacpy import mapper
from uacpy.output import serialization

DATA_DIR = Path("data")
TARGET_PATIENT_ID = "01"
SOURCE_PATIENT_IDS = ("02", "03", "04", "05", "06", "07")
UAC_SUBMESH_FILE_NAME = "uac_submesh_data.npz"
SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "mapped_fibers_to_patient_01.npz"
SNAPSHOT_SOURCE_IDS = ("02", "07")


def _uac_mapper(load_patient_mesh, patient_id: str):
    mesh = load_patient_mesh(patient_id)
    uac_submeshes = serialization.load_uac_submeshdata(
        DATA_DIR / f"patient_{patient_id}" / UAC_SUBMESH_FILE_NAME
    )
    return mesh, mapper.UACMapper(mesh, uac_submeshes)


@pytest.fixture(scope="module")
def target(load_patient_mesh):
    return _uac_mapper(load_patient_mesh, TARGET_PATIENT_ID)


def _map_fibers_to_target(load_patient_mesh, target_mapper, source_patient_ids):
    """Each source patient's fibers on the target mesh, keyed by `patient_<id>`."""
    mapped = {}
    for patient_id in source_patient_ids:
        source_mesh, source_mapper = _uac_mapper(load_patient_mesh, patient_id)
        source_uac_fibers = source_mapper.map_vector_from_original_to_uac_mesh(
            np.array(source_mesh.cell_data["fibers"])
        )
        target_uac_fibers = source_mapper.interpolate_to_other_uac_mesh(
            source_uac_fibers, target_mapper
        )
        mapped[f"patient_{patient_id}"] = target_mapper.map_vector_from_uac_to_original_mesh(
            target_uac_fibers, normalize=True
        )
    return mapped


@pytest.fixture(scope="module")
def mapped_fibers(load_patient_mesh, target):
    _, target_mapper = target
    return _map_fibers_to_target(load_patient_mesh, target_mapper, SOURCE_PATIENT_IDS)


@pytest.mark.slow
@pytest.mark.parametrize("patient_id", SOURCE_PATIENT_IDS)
def test_mapped_fibers_are_unit_tangent_vectors_of_the_target_mesh(
    target, mapped_fibers, patient_id
):
    """Every target cell is covered by a segment, and its mapped fiber has unit length and lies in
    the cell's tangent plane -- both hold by construction of the Jacobian mapping, whatever the
    source field was."""
    target_mesh, _ = target
    fibers = mapped_fibers[f"patient_{patient_id}"]
    lengths = np.linalg.norm(fibers, axis=1)

    assert (lengths > 0).all(), "some target cells are not covered by any segment"
    assert_allclose(lengths, 1.0, atol=1e-9)
    normals = np.array(
        target_mesh.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"]
    )
    assert_allclose(np.einsum("ij,ij->i", fibers, normals), 0.0, atol=1e-6)


@pytest.mark.slow
def test_mapped_fibers_match_snapshot(mapped_fibers):
    """The mapped fields of the snapshotted sources equal the pinned values; regenerate only for
    a deliberate change."""
    snapshot = np.load(SNAPSHOT_PATH)

    assert set(snapshot.files) == {f"patient_{i}" for i in SNAPSHOT_SOURCE_IDS}
    for key in snapshot.files:
        assert mapped_fibers[key].shape == snapshot[key].shape, f"{key}: regenerate the snapshot"
        assert_allclose(mapped_fibers[key], snapshot[key], atol=1e-5, err_msg=key)


def _regenerate_snapshot():
    import pyvista as pv

    from uacpy.common import mesh_utils

    def load_patient_mesh(patient_id):
        path = DATA_DIR / f"patient_{patient_id}" / "mesh_with_fibers_tags.vtu"
        return mesh_utils.convert_unstructured_to_polydata_mesh(pv.read(path))

    _, target_mapper = _uac_mapper(load_patient_mesh, TARGET_PATIENT_ID)
    mapped = _map_fibers_to_target(load_patient_mesh, target_mapper, SNAPSHOT_SOURCE_IDS)
    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        SNAPSHOT_PATH, **{key: value.astype(np.float32) for key, value in mapped.items()}
    )


if __name__ == "__main__":
    _regenerate_snapshot()

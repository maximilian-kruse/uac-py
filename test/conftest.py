"""Shared fixtures, and automatic `unit`/`integration` marking by directory.

Everything under `test/unit/` is marked `unit` and everything under `test/integration/` is marked
`integration`, so tests don't carry those markers themselves (`slow` is still applied by hand).
The real-patient fixtures are session-scoped: the full pipeline runs once however many
integration tests consume it.
"""

from pathlib import Path

import pytest
import pyvista as pv

from uacpy.anatomies import left_atrium_4_pvs as config
from uacpy.common import mesh_utils
from uacpy.construction import constructor

DATA_DIR = Path("data")
PATIENT_MESH_NAME = "mesh_with_fibers_tags.vtu"
ANATOMICAL_TAGS = {"MV": 0, "LAA": 1, "LIPV": 2, "LSPV": 3, "RIPV": 4, "RSPV": 5}


def pytest_collection_modifyitems(items):
    for item in items:
        parts = item.path.parts
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)


def _load_patient_mesh(patient_id: str) -> pv.PolyData:
    mesh = pv.read(DATA_DIR / f"patient_{patient_id}" / PATIENT_MESH_NAME)
    return mesh_utils.convert_unstructured_to_polydata_mesh(mesh)


def construct_uac_data(mesh: pv.PolyData):
    """Run the full LA construction pipeline on `mesh` and return its `UACData`."""
    settings = constructor.UACConstructorSettings(
        mesh=mesh,
        feature_tags=ANATOMICAL_TAGS,
        path_config=config.path_configs,
        marker_config=config.marker_configs,
        parameterization_config=config.parameterization_configs,
        uac_config=config.uac_configs,
        submesh_config=config.submesh_configs,
    )
    uac_constructor = constructor.UACConstructor(settings)
    uac_constructor.construct_segmentation()
    uac_constructor.construct_uacs()
    return uac_constructor.data


@pytest.fixture(scope="session")
def load_patient_mesh():
    """Function loading a patient's tagged surface mesh (with `fibers` cell data) by id, e.g.
    `"01"`, as a triangular `PolyData`. Test modules can't import from `conftest` (importlib
    mode), hence a fixture."""
    return _load_patient_mesh


@pytest.fixture(scope="session")
def patient_01_mesh() -> pv.PolyData:
    return _load_patient_mesh("01")


@pytest.fixture(scope="session")
def patient_01_uac_data(patient_01_mesh):
    return construct_uac_data(patient_01_mesh)

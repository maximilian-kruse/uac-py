"""Regression test for the la_construction pipeline against a golden snapshot.

The snapshot (per-segment vertex/cell counts and (alpha, beta) ranges) was captured by running
the pipeline once against data/patient_01/mesh_with_fibers_tags.vtu. A failure here means either
a real behavior change in the construction pipeline (investigate before touching this file) or a
deliberate change to the input mesh or the LA anatomy config (regenerate the snapshot, don't
just loosen the assertions to make it pass).
"""

import numpy as np
import pytest
import pyvista as pv

from uacpy.anatomies import left_atrium_4_pvs as config
from uacpy.common import mesh_utils
from uacpy.construction import check, constructor

MESH_PATH = "data/patient_01/mesh_with_fibers_tags.vtu"
ANATOMICAL_TAGS = {"MV": 0, "LAA": 1, "LIPV": 2, "LSPV": 3, "RIPV": 4, "RSPV": 5}

# (n_vertices, n_cells, alpha_min, alpha_max, beta_min, beta_max) per submesh, keyed as
# the flattened key sequences of `submesh_configs`.
SNAPSHOT = {
    ("LIPV", "segment_1"): (544, 1003, 1.8, 2.0, 1.3, 1.5),
    ("LIPV", "segment_2"): (470, 863, 1.8, 2.1414213562373097, 1.5, 1.6414213562373094),
    ("LIPV", "segment_3"): (291, 516, 2.0, 2.1414213562373097, 1.3, 1.6414213562373094),
    ("LSPV", "segment_1"): (155, 261, 1.8, 2.0, 1.0, 1.2),
    ("LSPV", "segment_2"): (117, 193, 1.8, 2.1414213562373097, 0.8585786437626904, 1.0),
    ("LSPV", "segment_3"): (154, 256, 2.0, 2.1414213562373097, 0.8585786437626904, 1.2),
    ("RSPV", "segment_1"): (240, 422, 1.0, 1.2, 1.0, 1.2),
    ("RSPV", "segment_2"): (175, 300, 0.8585786437626904, 1.2, 0.8585786437626906, 1.0),
    ("RSPV", "segment_3"): (353, 631, 0.8585786437626904, 1.0, 0.8585786437626906, 1.2),
    ("RIPV", "segment_1"): (167, 285, 1.0, 1.2, 1.3, 1.5),
    ("RIPV", "segment_2"): (554, 1028, 0.8585786437626906, 1.2, 1.5, 1.6414213562373094),
    ("RIPV", "segment_3"): (352, 627, 0.8585786437626906, 1.0, 1.3, 1.6414213562373094),
    ("roof",): (1116, 2055, 1.0, 2.0, 1.0, 1.5),
    ("anterior",): (2607, 5003, 0.8585786437626904, 2.1414213562373097, 0.0, 1.0),
    ("septal",): (2569, 4935, 0.0, 1.0, 0.8585786437626906, 1.6414213562373094),
    ("posterior_roof",): (
        1522,
        2863,
        0.8585786437626906,
        2.1414213562373097,
        1.5,
        2.1414213562373092,
    ),
    ("posterior_mv",): (769, 1423, 0.8585786437626906, 2.1414213562373097, 2.1414213562373092, 2.5),
    ("lateral_roof",): (758, 1385, 2.0, 2.6414213562373097, 0.8585786437626904, 1.6414213562373094),
    ("lateral_mv",): (544, 990, 2.6414213562373097, 3.0, 0.8585786437626904, 1.6414213562373094),
    ("laa",): (3219, 6336, 2.2, 3.0, 1.7000000000000002, 2.5),
}
MESH_POINT_COUNT = 15820
MESH_CELL_COUNT = 31375


@pytest.mark.unit
def test_segment_boundary_multiplicity():
    """Purely structural, on the config alone -- no mesh needed, so this runs fast."""
    assert check.check_segment_boundary_multiplicity(
        path_config=config.path_configs,
        marker_config=config.marker_configs,
        parameterization_config=config.parameterization_configs,
        uac_config=config.uac_configs,
        submesh_config=config.submesh_configs,
    )


@pytest.fixture(scope="module")
def uac_data():
    """The `UACData` from running the full LA pipeline once on the patient mesh."""
    mesh = pv.read(MESH_PATH)
    mesh = mesh_utils.convert_unstructured_to_polydata_mesh(mesh)
    assert mesh.n_points == MESH_POINT_COUNT, "input mesh changed -- regenerate the snapshot"
    assert mesh.n_cells == MESH_CELL_COUNT, "input mesh changed -- regenerate the snapshot"

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


@pytest.mark.integration
@pytest.mark.slow
def test_uac_mesh_is_valid(uac_data):
    """No UAC submesh has a folded-over cell."""
    assert check.check_uac_mesh_validity(uac_data.uac_submesh_data)


@pytest.mark.integration
@pytest.mark.slow
def test_uac_tiling(uac_data):
    """No two segments overlap in UAC space."""
    assert check.check_uac_tiling(uac_data.submesh_boundary_data)


@pytest.mark.integration
@pytest.mark.slow
def test_uac_mesh_matches_snapshot(uac_data):
    """Per-segment vertex/cell counts and (alpha, beta) ranges equal the pinned snapshot."""
    uac_submesh_data = uac_data.uac_submesh_data
    assert uac_submesh_data.keys() == SNAPSHOT.keys(), (
        "segment set changed -- regenerate the snapshot"
    )

    for key, (n_vertices, n_cells, alpha_min, alpha_max, beta_min, beta_max) in SNAPSHOT.items():
        submesh = uac_submesh_data[key]
        assert submesh.alpha.shape[0] == n_vertices, f"{key}: vertex count changed"
        assert submesh.connectivity.shape[0] == n_cells, f"{key}: cell count changed"
        np.testing.assert_allclose(
            submesh.alpha.min(), alpha_min, atol=1e-9, err_msg=f"{key}: alpha_min"
        )
        np.testing.assert_allclose(
            submesh.alpha.max(), alpha_max, atol=1e-9, err_msg=f"{key}: alpha_max"
        )
        np.testing.assert_allclose(
            submesh.beta.min(), beta_min, atol=1e-9, err_msg=f"{key}: beta_min"
        )
        np.testing.assert_allclose(
            submesh.beta.max(), beta_max, atol=1e-9, err_msg=f"{key}: beta_max"
        )

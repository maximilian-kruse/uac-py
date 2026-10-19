"""Unit tests for `check.py`'s three defect-detection branches, each on hand-built minimal data.

The golden-snapshot fixture in `test_la_construction.py` only ever exercises the "everything is
valid" path of these checks (the reference mesh has no defects by construction), so none of them
actually verify the detection logic itself. These tests build the smallest possible defect that
should trip each check, directly on plain dataclasses/dicts -- no mesh needed for any of them.
"""

import numpy as np
import pytest

from uacpy.construction import check, model


def test_uac_mesh_validity_detects_a_foldover():
    """A degenerate (collinear) triangle has zero area, which the fold-over check must reject."""
    degenerate_submesh = model.UACSubmesh(
        vertex_inds=np.array([0, 1, 2]),
        connectivity=np.array([[0, 1, 2]]),
        cell_inds=np.array([0]),
        alpha=np.array([0.0, 1.0, 2.0]),
        beta=np.array([0.0, 0.0, 0.0]),
    )

    assert not check.check_uac_mesh_validity({"seg": degenerate_submesh})


def test_uac_mesh_validity_accepts_a_regular_triangle():
    """The mirror-image case: a single non-degenerate triangle must pass."""
    valid_submesh = model.UACSubmesh(
        vertex_inds=np.array([0, 1, 2]),
        connectivity=np.array([[0, 1, 2]]),
        cell_inds=np.array([0]),
        alpha=np.array([0.0, 1.0, 0.0]),
        beta=np.array([0.0, 0.0, 1.0]),
    )

    assert check.check_uac_mesh_validity({"seg": valid_submesh})


def _single_boundary_configs(num_segments: int) -> dict:
    """The smallest config set with one boundary path and `num_segments` segments all referencing
    that path's entire (0, 1) portion -- so `num_segments` controls the resulting multiplicity.
    """
    path_config = {"p": model.BoundaryPath(feature_tag="X", coincides_with_mesh_boundary=True)}
    marker_config = {}
    parameterization_config = {
        "p": model.PathParameterization(path=["p"], markers=[], marker_relative_positions=[])
    }
    uac_config = {
        "p": model.UACMapping(path=["p"], relative_positions=[0, 1], uacs=[(0, 0), (1, 0)])
    }
    submesh_config = {
        f"s{i}": model.Segment(boundary_paths=[["p"]], portions=[(0, 1)], outside_path=["p"])
        for i in range(num_segments)
    }
    return {
        "path_config": path_config,
        "marker_config": marker_config,
        "parameterization_config": parameterization_config,
        "uac_config": uac_config,
        "submesh_config": submesh_config,
    }


def test_segment_boundary_multiplicity_detects_a_triple_reference():
    """A boundary portion referenced by 3 segments is neither an outer edge (1) nor a shared
    interior edge (2), so it must be flagged invalid.
    """
    assert not check.check_segment_boundary_multiplicity(**_single_boundary_configs(3))


@pytest.mark.parametrize("num_segments", [1, 2])
def test_segment_boundary_multiplicity_accepts_one_or_two_references(num_segments):
    """The two legitimate cases: an outer mesh edge (1 segment) or a shared interior edge (2)."""
    assert check.check_segment_boundary_multiplicity(**_single_boundary_configs(num_segments))


def _square_boundary(offset: float) -> model.UACPath:
    """A closed unit-square boundary loop, shifted by `offset` in both alpha and beta."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) + offset
    return model.UACPath(
        inds=np.arange(4),
        relative_lengths=np.linspace(0, 1, 4, endpoint=False),
        alpha=corners[:, 0],
        beta=corners[:, 1],
    )


def test_uac_tiling_detects_an_overlap():
    """Two unit squares offset by half a unit overlap in a 0.5x0.5 region -- must be rejected."""
    submesh_boundary_data = {"a": _square_boundary(0.0), "b": _square_boundary(0.5)}

    assert not check.check_uac_tiling(submesh_boundary_data)


def test_uac_tiling_accepts_disjoint_segments():
    """The mirror-image case: two unit squares offset by a full unit don't overlap at all."""
    submesh_boundary_data = {"a": _square_boundary(0.0), "b": _square_boundary(1.0)}

    assert check.check_uac_tiling(submesh_boundary_data)

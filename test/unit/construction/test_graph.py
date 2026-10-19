"""Unit tests for `construction/graph.link`: key-sequence references become live node references.

A one-entry-per-kind config stands in for a real anatomy; `link` never touches a mesh, so none is
needed.
"""

import pytest

from uacpy.construction import graph, model


def _minimal_configs(outside_path=("rim",)) -> dict:
    return {
        "path_config": {
            "rim": model.BoundaryPath(feature_tag="MV", coincides_with_mesh_boundary=True)
        },
        "marker_config": {
            "start": model.Marker(path=("rim",), position_type=model.PositionKind.INDEX, position=0)
        },
        "parameterization_config": {
            "rim": model.PathParameterization(
                path=("rim",), markers=[("start",)], marker_relative_positions=[0.0]
            )
        },
        "uac_config": {
            "rim": model.UACMapping(
                path=("rim",), relative_positions=[0.0, 1.0], uacs=[(0.0, 0.0), (1.0, 0.0)]
            )
        },
        "submesh_config": {
            "inside": model.Segment(
                boundary_paths=[("rim",)], portions=[(0.0, 1.0)], outside_path=outside_path
            )
        },
    }


def test_link_replaces_key_sequences_with_the_nodes_they_name():
    linked = graph.link(**_minimal_configs())

    marker = linked.markers[("start",)]
    segment = linked.segments[("inside",)]
    assert marker.source is linked.paths[("rim",)]
    assert marker.dependencies == [linked.paths[("rim",)]]
    assert segment.boundary_paths == [linked.uacs[("rim",)]]
    assert segment.outside_path is linked.paths[("rim",)]
    assert linked.uacs[("rim",)].parameterization is linked.parameterizations[("rim",)]


def test_link_raises_a_key_error_naming_a_dangling_reference():
    with pytest.raises(KeyError, match="Dangling reference"):
        graph.link(**_minimal_configs(outside_path=("does_not_exist",)))

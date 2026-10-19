"""Structural test of the LA anatomy config alone -- no mesh needed, so it runs fast and can catch a
bad config edit before the pipeline is ever run."""

from uacpy.anatomies import left_atrium_4_pvs as config
from uacpy.construction import check


def test_segment_boundary_multiplicity():
    """Purely structural, on the config alone -- no mesh needed, so this runs fast."""
    assert check.check_segment_boundary_multiplicity(
        path_config=config.path_configs,
        marker_config=config.marker_configs,
        parameterization_config=config.parameterization_configs,
        uac_config=config.uac_configs,
        submesh_config=config.submesh_configs,
    )

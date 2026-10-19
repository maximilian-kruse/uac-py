"""Unit test for `serialization.py`'s `.npz` round trip.

Pure data in/out -- no mesh or pipeline run needed, just hand-built `UACSubmesh`es in the same
flat keyed shape `UACConstructor.data.uac_submesh_data` would produce.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from uacpy.construction import model
from uacpy.output import serialization


def _submesh(offset: float = 0.0) -> model.UACSubmesh:
    return model.UACSubmesh(
        vertex_inds=np.array([0, 1, 2, 3]),
        connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        cell_inds=np.array([10, 11]),
        alpha=np.array([0.0, 1.0, 1.0, 0.0]) + offset,
        beta=np.array([0.0, 0.0, 1.0, 1.0]) + offset,
    )


@pytest.mark.unit
def test_uac_submeshdata_round_trips_exactly_through_a_file(tmp_path):
    """Saving then loading reproduces every field of every keyed `UACSubmesh` exactly."""
    submesh_data = {("LIPV", "segment_1"): _submesh(), ("roof",): _submesh(2.0)}
    path = tmp_path / "submesh.npz"

    serialization.save_uac_submeshdata(path, submesh_data)
    loaded = serialization.load_uac_submeshdata(path)

    assert loaded.keys() == submesh_data.keys()
    for key, original in submesh_data.items():
        assert isinstance(loaded[key], model.UACSubmesh)
        for field in ("vertex_inds", "connectivity", "cell_inds", "alpha", "beta"):
            assert_allclose(getattr(loaded[key], field), getattr(original, field))
            assert getattr(loaded[key], field).dtype == getattr(original, field).dtype

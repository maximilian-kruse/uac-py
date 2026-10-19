"""Unit tests for `mesh_utils.py`'s tag-defect-fixup pipeline, each on the smallest synthetic
mesh that reproduces one defect.

`preprocess_mesh` itself (decimation/smoothing/fiber interpolation) is only exercised indirectly,
through the golden-snapshot fixture in `test_la_construction.py` -- it needs real mesh geometry to
be meaningful. The four `_fix_*` stages `fix_feature_tag_defects` chains together are each a
self-contained graph-connectivity operation on `anatomical_tags` cell data, so each gets its own
hand-built mesh with exactly the one defect it's meant to catch, tagged directly rather than
derived from any real geometry.
"""

import numpy as np
import pyvista as pv
from numpy.testing import assert_array_equal

from uacpy.common import mesh_utils


def _build_closed_hexagon_fan() -> tuple[np.ndarray, list[list[int]]]:
    """A flat hexagon triangulated as 6 triangles fanning out from its center.

    Every fan triangle has exactly 2 neighbors (its two spoke edges; the outer perimeter edge is a
    genuine mesh boundary) -- a stable "body" shape to attach a single spike triangle to, without
    the body itself ever being mistaken for a spike.

    Returns `(points, fan_faces)`: `points[0]` is the center, `points[1:7]` the ring, in travel
    order; `fan_faces[i]` is `[center, ring[i], ring[i + 1]]`.
    """
    angles = np.deg2rad(np.arange(6) * 60)
    ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    center = np.array([[0.0, 0.0, 0.0]])
    points = np.vstack([center, ring])
    fan_faces = [[0, i + 1, (i + 1) % 6 + 1] for i in range(6)]
    return points, fan_faces


def test_fix_feature_tag_defects_chains_all_four_stages():
    """A feature-tag spike (untagged by stage 2), once untagged, is now a plain boundary spike
    (unrelated to any tag) that stage 4 must go on to remove entirely -- exercising the actual
    chaining `fix_feature_tag_defects` exists for, not just each stage in isolation.
    """
    points, fan_faces = _build_closed_hexagon_fan()
    spike_point = np.array([[2.5, 0.5, 0.0]])
    points = np.vstack([points, spike_point])
    spike_face = [1, 2, 7]
    mesh = pv.PolyData.from_regular_faces(points, np.array([*fan_faces, spike_face]))
    mesh.cell_data["anatomical_tags"] = np.array([1] * 7, dtype=int)

    fixed = mesh_utils.fix_feature_tag_defects(mesh)

    assert fixed.n_cells == 6
    assert fixed.n_points == 7
    assert_array_equal(fixed.cell_data["anatomical_tags"], [1] * 6)


def test_fix_boundary_spikes_removes_a_single_neighbor_cell_and_its_orphaned_point():
    """A triangle attached to the fan by only one edge (its other vertex is its own, unshared
    point) has exactly 1 neighbor mesh-wide, regardless of tagging -- `_fix_boundary_spikes` must
    drop the cell and then compact away that now-unused point.
    """
    points, fan_faces = _build_closed_hexagon_fan()
    spike_point = np.array([[2.5, 0.5, 0.0]])
    points = np.vstack([points, spike_point])
    spike_face = [1, 2, 7]  # shares only edge (1, 2) with fan_faces[0]
    mesh = pv.PolyData.from_regular_faces(points, np.array([*fan_faces, spike_face]))

    fixed = mesh_utils._fix_boundary_spikes(mesh)

    assert fixed.n_cells == 6
    assert fixed.n_points == 7


def test_fix_feature_tag_dangles_keeps_only_the_largest_connected_component():
    """Two geometrically disjoint patches sharing the same tag: the smaller one is a "dangle" that
    must be untagged, the larger one (the real feature) must keep its tag.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],  # main blob (2 triangles)
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
            [10.0, 11.0, 0.0],  # dangle (1 triangle), disjoint vertices
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    mesh = pv.PolyData.from_regular_faces(points, faces)
    mesh.cell_data["anatomical_tags"] = np.array([1, 1, 1], dtype=int)

    fixed = mesh_utils._fix_feature_tag_dangles(mesh)

    assert_array_equal(fixed.cell_data["anatomical_tags"], [1, 1, 0])


def test_fix_feature_tag_spikes_untags_a_single_neighbor_cell_within_its_own_tag():
    """The same fan-plus-spike shape as the boundary-spike test, but now entirely tagged as one
    feature: the spike (1 neighbor within the tag's own region) must be untagged, the fan must not.
    """
    points, fan_faces = _build_closed_hexagon_fan()
    spike_point = np.array([[2.5, 0.5, 0.0]])
    points = np.vstack([points, spike_point])
    spike_face = [1, 2, 7]
    mesh = pv.PolyData.from_regular_faces(points, np.array([*fan_faces, spike_face]))
    mesh.cell_data["anatomical_tags"] = np.array([1] * 7, dtype=int)

    fixed = mesh_utils._fix_feature_tag_spikes(mesh)

    assert_array_equal(fixed.cell_data["anatomical_tags"], [1, 1, 1, 1, 1, 1, 0])


def test_fix_body_tag_spikes_reassigns_a_single_neighbor_untagged_cell_to_its_surrounding_tag():
    """The untagged-body mirror of the feature-tag-spike case: a tag-0 spike attached to a tag-0
    fan by one edge, but whose other two edges border a tag-1 region, must be reassigned to tag 1
    (the tag that actually surrounds it) rather than merely dropped.
    """
    points, fan_faces = _build_closed_hexagon_fan()
    q, x, y = np.array([[2.0, 0.3, 0.0]]), np.array([[3.0, 1.0, 0.0]]), np.array([[3.0, -0.5, 0.0]])
    points = np.vstack([points, q, x, y])
    spike_face = [1, 2, 7]  # (ring[0], ring[1], Q) -- shares edge (1, 2) with the fan
    feature_face_one = [2, 7, 8]  # (ring[1], Q, X)
    feature_face_two = [7, 1, 9]  # (Q, ring[0], Y)
    faces = np.array([*fan_faces, spike_face, feature_face_one, feature_face_two])
    mesh = pv.PolyData.from_regular_faces(points, faces)
    mesh.cell_data["anatomical_tags"] = np.array([0] * 6 + [0, 1, 1], dtype=int)

    fixed = mesh_utils._fix_body_tag_spikes(mesh)

    assert_array_equal(fixed.cell_data["anatomical_tags"], [0, 0, 0, 0, 0, 0, 1, 1, 1])

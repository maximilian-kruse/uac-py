"""Unit tests for `mapper.UACMapper`'s field transfer, against tiny synthetic meshes.

No real mesh fixture needed: builds one segment directly as an `model.UACSubmesh` (four
vertices, two triangles) rather than running the full construction pipeline, since the invariants
under test hold for any non-degenerate triangle, not just real anatomy: original-mesh <-> UAC-space
round-tripping (exact, since both directions of a single mapper's own gather/scatter and Jacobian
conversion are each other's inverse) and cross-patient interpolation (exact only for a constant
field, since resampling between two different triangulations is otherwise approximate).
"""

import numpy as np
import pytest
import pyvista as pv
from numpy.testing import assert_allclose

from uacpy import mapper
from uacpy.construction import model


def _original_mesh(vertices: np.ndarray, connectivity: np.ndarray) -> pv.PolyData:
    """The patient's original triangle mesh, whose cells the single segment covers entirely."""
    return pv.PolyData.from_regular_faces(vertices, connectivity)


def _build_single_segment_mapper() -> mapper.UACMapper:
    """A mapper over one two-triangle segment that is the entire "original mesh"."""
    original_mesh_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.3],
            [1.0, 1.0, 0.1],
            [0.0, 1.0, -0.2],
        ]
    )
    uac_submesh = model.UACSubmesh(
        vertex_inds=np.array([0, 1, 2, 3]),
        connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        cell_inds=np.array([0, 1]),
        alpha=np.array([0.0, 1.0, 1.0, 0.0]),
        beta=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    return mapper.UACMapper(
        _original_mesh(original_mesh_vertices, uac_submesh.connectivity),
        {("seg",): uac_submesh},
    )


@pytest.mark.unit
def test_scalar_field_round_trips_exactly_through_uac_space():
    """original -> UAC -> original must be the identity: `cell_inds` is a bijection here (one
    segment covering the whole mesh, no overlap), so the gather/scatter is exact, not approximate.
    """
    uac_mapper = _build_single_segment_mapper()
    rng = np.random.default_rng(0)
    original_scalar_field = rng.uniform(-1, 1, size=2)

    uac_scalar_data = uac_mapper.map_scalar_from_original_to_uac_mesh(original_scalar_field)
    roundtrip = uac_mapper.map_scalar_from_uac_to_original_mesh(uac_scalar_data)

    assert_allclose(roundtrip, original_scalar_field)


def _build_mapper_over_shared_uac_domain(original_mesh_vertices: np.ndarray) -> mapper.UACMapper:
    """A mapper over the same two-triangle UAC domain as `_build_single_segment_mapper`, but with
    its own (distinct) `original_mesh_vertices` -- standing in for a different patient's mesh that
    happens to share this segment's (alpha, beta) target domain, as any two patients' UAC atlases
    do by construction.
    """
    uac_submesh = model.UACSubmesh(
        vertex_inds=np.array([0, 1, 2, 3]),
        connectivity=np.array([[0, 1, 2], [0, 2, 3]]),
        cell_inds=np.array([0, 1]),
        alpha=np.array([0.0, 1.0, 1.0, 0.0]),
        beta=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    return mapper.UACMapper(
        _original_mesh(original_mesh_vertices, uac_submesh.connectivity),
        {("seg",): uac_submesh},
    )


@pytest.mark.unit
def test_interpolate_to_other_uac_mesh_reproduces_a_constant_field():
    """A constant scalar field, resampled onto a different patient's mapper that shares this
    segment's UAC domain, must come back exactly that same constant everywhere -- interpolating a
    constant is the identity regardless of how the two patients' original meshes (and therefore
    Jacobians) actually differ, since `interpolate_to_other_uac_mesh` only ever looks at the shared
    (alpha, beta) domain, never at `original_mesh_vertices`.
    """
    mapper_a = _build_mapper_over_shared_uac_domain(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.3], [1.0, 1.0, 0.1], [0.0, 1.0, -0.2]])
    )
    mapper_b = _build_mapper_over_shared_uac_domain(
        np.array([[0.0, 0.0, 1.0], [1.1, 0.0, 0.0], [1.0, 0.9, 0.4], [0.1, 1.0, 0.0]])
    )
    constant_field = {("seg",): np.array([5.0, 5.0])}

    transferred = mapper_a.interpolate_to_other_uac_mesh(constant_field, mapper_b)

    assert_allclose(transferred[("seg",)], [5.0, 5.0])


@pytest.mark.unit
def test_vector_field_round_trips_exactly_through_uac_space():
    """UAC -> original -> UAC must be the identity: mapping a 2D UAC vector into 3D applies the
    per-cell Jacobian `J` (embedding the tangent plane), and mapping back applies its Moore-Penrose
    pseudoinverse `J+`; `J+ @ J = I` exactly for any Jacobian with full column rank (i.e. any
    non-degenerate triangle), so this round trip is exact regardless of the triangle's own shape.
    """
    uac_mapper = _build_single_segment_mapper()
    rng = np.random.default_rng(1)
    uac_vector_field = rng.uniform(-1, 1, size=(2, 2))

    original_vector_field = uac_mapper.map_vector_from_uac_to_original_mesh(
        {("seg",): uac_vector_field}
    )
    roundtrip = uac_mapper.map_vector_from_original_to_uac_mesh(original_vector_field)

    assert_allclose(roundtrip[("seg",)], uac_vector_field, atol=1e-9)


def _build_mapper_from_triangulation(
    original_mesh_vertices: np.ndarray,
    alpha_beta: np.ndarray,
    connectivity: np.ndarray,
) -> mapper.UACMapper:
    """A mapper over one segment with the given triangulation of the (alpha, beta) domain."""
    num_vertices = original_mesh_vertices.shape[0]
    uac_submesh = model.UACSubmesh(
        vertex_inds=np.arange(num_vertices),
        connectivity=connectivity,
        cell_inds=np.arange(connectivity.shape[0]),
        alpha=alpha_beta[:, 0],
        beta=alpha_beta[:, 1],
    )
    return mapper.UACMapper(
        _original_mesh(original_mesh_vertices, connectivity), {("seg",): uac_submesh}
    )


@pytest.mark.unit
def test_nearest_transfer_onto_itself_is_exact():
    """The centroid of every cell lies in that very cell of the same mesh, so a field transferred
    onto its own mesh comes back unchanged."""
    uac_mapper = _build_single_segment_mapper()
    field = {("seg",): np.array([[1.0, 0.0], [0.0, -2.0]])}

    transferred = uac_mapper.interpolate_to_other_uac_mesh(field, uac_mapper)

    assert_allclose(transferred[("seg",)], field[("seg",)])


@pytest.mark.unit
def test_nearest_transfer_takes_the_value_of_the_containing_cell():
    """The source is the unit square cut along its diagonal, the target the same square cut into
    four triangles around its center. The centroids of the two target triangles below the
    diagonal lie in the source triangle below it, the two above in the one above it."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    source = _build_mapper_from_triangulation(
        np.hstack([square, np.zeros((4, 1))]), square, np.array([[0, 1, 2], [0, 2, 3]])
    )
    center_alpha_beta = np.vstack([square, [0.5, 0.5]])
    target = _build_mapper_from_triangulation(
        np.hstack([center_alpha_beta, np.zeros((5, 1))]),
        center_alpha_beta,
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    )

    transferred = source.interpolate_to_other_uac_mesh({("seg",): np.array([10.0, 20.0])}, target)

    assert_allclose(transferred[("seg",)], [10.0, 10.0, 20.0, 20.0])


@pytest.mark.unit
def test_transfer_keeps_the_sign_and_length_of_vectors():
    """Nearest-neighbor transfer copies vectors as they are, so opposite fibers of one axis stay
    opposite and keep their lengths (nothing is averaged, so nothing cancels)."""
    mapper_a = _build_mapper_over_shared_uac_domain(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.3], [1.0, 1.0, 0.1], [0.0, 1.0, -0.2]])
    )
    mapper_b = _build_mapper_over_shared_uac_domain(
        np.array([[0.0, 0.0, 1.0], [1.1, 0.0, 0.0], [1.0, 0.9, 0.4], [0.1, 1.0, 0.0]])
    )
    direction = np.array([np.cos(0.4), np.sin(0.4)])
    field = {("seg",): np.array([3.0 * direction, -0.5 * direction])}

    transferred = mapper_a.interpolate_to_other_uac_mesh(field, mapper_b)

    assert_allclose(transferred[("seg",)], field[("seg",)])


@pytest.mark.unit
def test_normalized_mapping_to_the_original_mesh_gives_unit_vectors_of_the_same_direction():
    """`normalize=True` only rescales: the mapped vectors are the unit vectors of the unnormalized
    ones, however much the Jacobian of a cell stretches them."""
    uac_mapper = _build_single_segment_mapper()
    uac_vector_field = {("seg",): np.array([[5.0, 0.0], [-1.0, 2.0]])}

    mapped = uac_mapper.map_vector_from_uac_to_original_mesh(uac_vector_field)
    mapped_normalized = uac_mapper.map_vector_from_uac_to_original_mesh(
        uac_vector_field, normalize=True
    )

    assert_allclose(np.linalg.norm(mapped_normalized, axis=1), 1.0)
    assert_allclose(mapped_normalized, mapped / np.linalg.norm(mapped, axis=1, keepdims=True))

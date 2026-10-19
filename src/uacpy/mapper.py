"""Cross-patient/cross-representation field transfer: original mesh <-> UAC submeshes <-> another
patient's UAC submeshes.

A `UACMapper` wraps one patient's already-constructed UAC submeshes (from `constructor.
UACConstructor.data`) and caches the per-segment data (flattened pyvista submeshes, per-cell
Jacobians) that transfer needs, so those don't get recomputed on every call. Scalars move by
gather/scatter on `cell_inds` (a cell keeps its value, just addressed by a different index space);
vectors additionally change basis through each cell's own Jacobian, since a UAC-space vector and
its original-mesh tangent-space counterpart are related by that cell's local embedding, not the
identity.

Transfer between two patients is a nearest-neighbor copy in the shared (alpha, beta) domain: each
cell of the target takes the value of the source cell closest to its centroid. Nothing is averaged,
so the roughness of a field is kept (which statistics of a field, e.g. variance or correlation
length, need), a mapping of a mesh onto itself is exact, and axial fields such as fiber directions
need no special treatment (a copied vector keeps its length and sign). If the two meshes differ in
resolution, a source cell is repeated or skipped.
"""

import time
from dataclasses import dataclass
from typing import Self

import numpy as np
import pyvista as pv

from uacpy.common import dict_utils, jacobian
from uacpy.construction import model
from uacpy.output import logging

# ==================================================================================================
# One array per segment (per-cell values), keyed like `UACData.uac_submesh_data`: scalars of shape
# `(n_segment_cells,)` or vectors of shape `(n_segment_cells, k)`.
type FieldDict = dict_utils.KeySequenceDict[np.ndarray]


# ==================================================================================================
@dataclass(frozen=True)
class _SegmentCache:
    """What field transfer needs from one segment, computed once.

    Attributes:
        cell_inds: The segment's cells as indices into the original mesh, shape `(n_cells,)`.
        jacobians: Per-cell Jacobians mapping 2D UAC vectors to 3D original-mesh tangent vectors,
            shape `(n_cells, 3, 2)`.
        pinv_jacobians: Their pseudoinverses, for the opposite direction, shape `(n_cells, 2, 3)`.
        flat_mesh: The segment as a z=0 pyvista mesh in (alpha, beta) space, for the spatial
            lookup of `UACMapper.interpolate_to_other_uac_mesh`.
    """

    cell_inds: np.ndarray
    jacobians: np.ndarray
    pinv_jacobians: np.ndarray
    flat_mesh: pv.PolyData


# --------------------------------------------------------------------------------------------------
def _build_segment_cache(
    original_mesh_vertices: np.ndarray, submesh: model.UACSubmesh
) -> _SegmentCache:
    """Compute a segment's Jacobians (original-mesh tangent space <-> UAC space) and flat mesh.

    Both Jacobians come from the segment's own vertex positions in both spaces.
    """
    vertices_uac = np.hstack([submesh.alpha[:, None], submesh.beta[:, None]])
    jacobians = jacobian.compute_cellwise_jacobians(
        original_mesh_vertices[submesh.vertex_inds], vertices_uac, submesh.connectivity
    )
    return _SegmentCache(
        cell_inds=submesh.cell_inds,
        jacobians=jacobians,
        pinv_jacobians=np.linalg.pinv(jacobians),
        flat_mesh=submesh.to_polydata(),
    )


# ==================================================================================================
class UACMapper:
    """Caches one patient's per-segment Jacobians and flat meshes for repeated field transfer.

    Both are expensive to recompute per call, cheap to keep once built.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        original_mesh: pv.PolyData,
        uac_submesh_data: dict_utils.KeySequenceDict[model.UACSubmesh],
        logger: logging.BaseLogger | None = None,
    ):
        """Cache the Jacobians and flat meshes for one patient's UAC submeshes.

        Args:
            original_mesh: The patient's original triangle mesh, in the same units used to
                construct `uac_submesh_data`. Its cell count sets the size of fields mapped back
                onto it.
            uac_submesh_data: `UACSubmesh` per segment (`UACConstructor.data.uac_submesh_data`).
            logger: Receives debug messages. Defaults to discarding them.
        """
        self._logger = logging.resolve_logger(logger)
        self._num_cells = original_mesh.n_cells
        self._segments = {}
        for key, submesh in uac_submesh_data.items():
            start = time.perf_counter()
            self._segments[key] = _build_segment_cache(original_mesh.points, submesh)
            self._logger.debug(
                f"Cached segment {'.'.join(map(str, key))}: {len(submesh.vertex_inds)} vertices, "
                f"{len(submesh.cell_inds)} cells, {time.perf_counter() - start:.3f} s"
            )

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_original_to_uac_mesh(self, scalar_field: np.ndarray) -> FieldDict:
        """Gather an original-mesh scalar field onto each segment's cells.

        Args:
            scalar_field: One value per original-mesh cell, shape `(n_cells,)`.

        Returns:
            Dict with one array of shape `(n_segment_cells,)` per segment.
        """
        self._logger.debug(f"Gathering scalar field of shape {scalar_field.shape} onto segments")
        return {key: scalar_field[segment.cell_inds] for key, segment in self._segments.items()}

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_uac_to_original_mesh(self, uac_scalar_data: FieldDict) -> np.ndarray:
        """Scatter each segment's per-cell values back to their original-mesh cell indices.

        Args:
            uac_scalar_data: Dict with one array of shape `(n_segment_cells,)` per segment.

        Returns:
            Array of shape `(n_cells,)`. Cells not covered by any segment are zero.
        """
        return self._scatter(uac_scalar_data, trailing_shape=())

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_original_to_uac_mesh(self, vector_field: np.ndarray) -> FieldDict:
        """Gather an original-mesh vector field onto each segment, converted into UAC space.

        Each cell's vector is converted from the original mesh's own tangent space into 2D UAC
        space via that cell's pseudoinverse Jacobian.

        Args:
            vector_field: One 3D vector per original-mesh cell, shape `(n_cells, 3)`.

        Returns:
            Dict with one array of shape `(n_segment_cells, 2)` per segment.
        """
        self._logger.debug(f"Gathering vector field of shape {vector_field.shape} onto segments")
        return {
            key: np.einsum("ijk,ik->ij", segment.pinv_jacobians, vector_field[segment.cell_inds])
            for key, segment in self._segments.items()
        }

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_uac_to_original_mesh(
        self, uac_vector_data: FieldDict, normalize: bool = False
    ) -> np.ndarray:
        """Scatter each segment's UAC-space vectors back to original-mesh tangent vectors.

        Converts via each cell's own (non-pseudo-inverse) Jacobian -- the exact inverse of
        `map_vector_from_original_to_uac_mesh`'s conversion for vectors in the tangent plane,
        since `pinv(J) @ J = I` for a Jacobian with full column rank.

        Args:
            uac_vector_data: Dict with one array of shape `(n_segment_cells, 2)` per segment.
            normalize: Whether to scale every mapped vector to unit length. The length of a mapped
                vector only reflects how much the UAC chart stretches the cell (the largest
                singular value of its Jacobian), so for a direction field, e.g. fibers, the length
                carries no meaning. Defaults to `False`.

        Returns:
            Array of shape `(n_cells, 3)`. Cells not covered by any segment are zero.
        """
        original_mesh_data = self._scatter(
            {
                key: np.einsum("ijk,ik->ij", segment.jacobians, uac_vector_data[key])
                for key, segment in self._segments.items()
            },
            trailing_shape=(3,),
        )
        if normalize:
            lengths = np.linalg.norm(original_mesh_data, axis=1, keepdims=True)
            original_mesh_data = np.divide(
                original_mesh_data,
                lengths,
                out=np.zeros_like(original_mesh_data),
                where=lengths > 0,
            )
        return original_mesh_data

    # ----------------------------------------------------------------------------------------------
    def interpolate_to_other_uac_mesh(
        self, uac_data: FieldDict, other_uac_mapper: Self
    ) -> FieldDict:
        """Copy `uac_data` (this mapper's own UAC-space field) onto `other_uac_mapper`'s mesh.

        Both mappers' UAC submeshes for the same segment key share the same (alpha, beta) target
        domain but not the same triangulation (they come from different patients' original
        meshes), so transfer is a spatial lookup in that shared 2D domain, not a gather/scatter by
        matching indices. Each target cell takes the value of the source cell closest to its
        centroid (the cell containing the centroid, if there is one). Nothing is averaged: scalars
        and vectors, including axial ones such as fibers, are copied as they are.

        Args:
            uac_data: Dict with one array of per-cell values for each segment of this
                mapper, scalars of shape `(n_cells,)` or vectors of shape `(n_cells, k)`.
            other_uac_mapper: Mapper of the target patient. Must contain the same segment keys.

        Returns:
            Dict with one array of per-cell values per segment of `other_uac_mapper`.
        """
        transferred = {}
        for key, segment in self._segments.items():
            centroids = other_uac_mapper._segments[key].flat_mesh.cell_centers().points
            nearest_cells = segment.flat_mesh.find_closest_cell(centroids)
            self._logger.debug(
                f"Transfer {'.'.join(map(str, key))}: {len(centroids)} target cells from "
                f"{segment.flat_mesh.n_cells} source cells, "
                f"{np.unique(nearest_cells).size} distinct source cells used"
            )
            transferred[key] = uac_data[key][nearest_cells]
        return transferred

    # ----------------------------------------------------------------------------------------------
    def _scatter(self, per_segment_data: FieldDict, trailing_shape: tuple[int, ...]) -> np.ndarray:
        """Write each segment's per-cell values to its cells in an original-mesh array of zeros."""
        original_mesh_data = np.zeros((self._num_cells, *trailing_shape))
        covered = np.zeros(self._num_cells, dtype=bool)
        num_non_finite = 0
        for key, segment in self._segments.items():
            original_mesh_data[segment.cell_inds] = per_segment_data[key]
            covered[segment.cell_inds] = True
            num_non_finite += np.count_nonzero(~np.isfinite(per_segment_data[key]))
        self._logger.debug(
            f"Scattered {len(self._segments)} segments onto {self._num_cells} cells: "
            f"{np.count_nonzero(~covered)} cells uncovered, {num_non_finite} non-finite values"
        )
        return original_mesh_data

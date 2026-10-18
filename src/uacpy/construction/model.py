"""Data model of the construction pipeline: authored config and computed results.

The first half is the *authored* data an anatomy config is written in: pure-data cell-complex
dataclasses addressed by key sequence. The second half is the *computed* data the pipeline produces
from it: paths and submeshes with their (alpha, beta) UAC coordinates. Vertex and cell indices in
the results are into the original mesh unless a docstring says otherwise.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pyvista as pv

from uacpy.common.dict_utils import KeySequence


# --------------------------------------------------------------------------------------------------
class EndpointKind(StrEnum):
    """What one end of a `ConnectionPath` names: a whole path or a single marker."""

    PATH = "path"
    MARKER = "marker"


class PositionKind(StrEnum):
    """How a `Marker`'s `position` is read: a raw vertex index or a relative arclength."""

    INDEX = "index"
    RELATIVE = "relative"


# ==================================================================================================
# Provided Config
# ==================================================================================================


@dataclass
class Marker:
    """A single mesh vertex, located on an existing path by index or relative position.

    Attributes:
        path: Key sequence of the path the marker lies on: a raw path for
            `PositionKind.INDEX`, a parameterization for `PositionKind.RELATIVE`.
        position_type: `INDEX` for a position in the raw path's vertex array (negative values
            count from the end), `RELATIVE` for a relative arclength in `[0, 1]`.
        position: The index or relative length, according to `position_type`.
    """

    path: KeySequence
    position_type: PositionKind
    position: int | float


@dataclass
class BoundaryPath:
    """Vertex set describing boundary of a tagged region<.

    Attributes:
        feature_tag: Name of the tag whose region's boundary loop is extracted. The mapping to an
            integer value is given by `UACConstructorSettings.feature_tags`.
        coincides_with_mesh_boundary: Whether the loop is the part of the region's boundary lying
            on the mesh's own boundary (`True`) or the part in the mesh interior (`False`).
    """

    feature_tag: str
    coincides_with_mesh_boundary: bool


@dataclass
class ConnectionPath:
    """Shortest admissible path between two boundary sets.

    Attributes:
        boundary_types: For `start` and `end` in turn, whether it names a path (any of its
            vertices) or a marker (a single vertex).
        start: Key sequence of the start path or marker.
        end: Key sequence of the end path or marker.
        inadmissible_contact: Key sequences of paths whose vertices the new path may not touch,
            or `None` for none.
        inadmissible_along: Key sequences of paths the new path may touch but not travel along.
    """

    boundary_types: Sequence[EndpointKind]
    start: KeySequence
    end: KeySequence
    inadmissible_contact: Sequence[KeySequence] | None
    inadmissible_along: Sequence[KeySequence]


@dataclass
class PathParameterization:
    """An arclength parameterization of a path.

    Attributes:
        path: Key sequence of the raw path being parameterized.
        markers: Key sequences of the markers pinned to prescribed relative lengths.
        marker_relative_positions: Relative length in `[0, 1]` of each marker, in the same order
            as `markers`.
    """

    path: KeySequence
    markers: Sequence[KeySequence]
    marker_relative_positions: Sequence[float]


@dataclass
class UACMapping:
    """Target (alpha, beta) coordinates assigned along a `PathParameterization`.

    Attributes:
        path: Key sequence of the `PathParameterization` this maps.
        relative_positions: Increasing relative-length breakpoints in `[0, 1]`.
        uacs: Target `(alpha, beta)` at each breakpoint; linear in between.
    """

    path: KeySequence
    relative_positions: Sequence[float]
    uacs: Sequence[tuple[float, float]]


@dataclass
class Segment:
    """Submesh region bounded by portions of paths that constitute boundary.

    Attributes:
        boundary_paths: Key sequences of the `UACMapping`s forming the region's boundary.
        portions: `(start, end)` relative-length range of each entry of `boundary_paths` that
            belongs to this region.
        outside_path: Key sequence of a raw path lying outside the region, used to tell the
            inside of the boundary loop from the outside.
    """

    boundary_paths: Sequence[KeySequence]
    portions: Sequence[tuple[float, float]]
    outside_path: KeySequence


# ==================================================================================================
# Parameterization results
# ==================================================================================================


@dataclass
class ParameterizedPath:
    """Vertices of a connected path, plus each vertex's arclength position, in `[0, 1]` of the path.

    Attributes:
        inds: Vertex indices into the original mesh, ordered along the path, shape `(n,)`.
        relative_lengths: Relative arclength of each vertex in `[0, 1]`, shape `(n,)`.
    """

    inds: np.ndarray = None
    relative_lengths: np.ndarray = None


@dataclass
class UACPath(ParameterizedPath):
    """A `ParameterizedPath` with target (alpha, beta) UAC coordinates assigned at each vertex.

    Attributes:
        alpha: Target alpha coordinate of each vertex, shape `(n,)`.
        beta: Target beta coordinate of each vertex, shape `(n,)`.
    """

    alpha: np.ndarray = None
    beta: np.ndarray = None


@dataclass
class Submesh:
    """A sub-region of the original mesh: its vertex/cell indices and local connectivity.

    Attributes:
        vertex_inds: Indices of the region's vertices into the *original* mesh, shape
            `(n_vertices,)`.
        cell_inds: Indices of the region's cells into the *original* mesh, shape `(n_cells,)`.
        connectivity: Triangle vertex indices *local* to `vertex_inds` (i.e. into
            `0..len(vertex_inds)`, not the original mesh), shape `(n_cells, 3)`
    """

    vertex_inds: np.ndarray = None
    cell_inds: np.ndarray = None
    connectivity: np.ndarray = None


@dataclass
class UACSubmesh(Submesh):
    """A `Submesh` with (alpha, beta) UAC coordinates assigned at every vertex.

    Attributes:
        alpha: Alpha coordinate of each vertex, shape `(n_vertices,)`.
        beta: Beta coordinate of each vertex, shape `(n_vertices,)`.
    """

    alpha: np.ndarray = None
    beta: np.ndarray = None

    def to_polydata(self) -> pv.PolyData:
        """Flatten into a z=0 pyvista mesh in (alpha, beta) space, for checks/plots.

        Returns:
            Triangle mesh with points `(alpha, beta, 0)` and the submesh's local connectivity.
        """
        vertex_coordinates = np.hstack(
            [self.alpha[:, None], self.beta[:, None], np.zeros((self.alpha.shape[0], 1))]
        )
        return pv.PolyData.from_regular_faces(vertex_coordinates, self.connectivity)

"""Resolves a construction config's key-sequence references into a graph of live object nodes.

`model.py`'s config dataclasses stay pure data, addressed by key-sequence, so an anatomy config
can still be authored in whatever order reads best (see e.g.
`uacpy.anatomies.left_atrium_4_pvs`, which groups the PVs together before the
roof/mitral connectors those PVs' own markers actually depend on -- authoring order and
dependency order genuinely differ here). `link` resolves that
config into a graph where every key-sequence field has become a direct object reference, once, at
construction time -- so a dangling reference is a `KeyError` here instead of wherever the pipeline
first reaches it.

Each `Node` type also owns its own `construct*` method(s) -- the logic for turning that one node's
config plus its (already-resolved) dependencies' results into its own result. `UACConstructor`
only orchestrates: it topologically sorts the graph and calls `node.construct(...)` in that order,
it doesn't know *how* any individual node is computed. That split is deliberate: without it,
`UACConstructor` would be both the thing that decides what order to do work in and the thing that
implements every kind of work, for five different node types -- a god class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from uacpy.common import dict_utils, mvc
from uacpy.construction import geometry, model
from uacpy.construction.model import EndpointKind, PositionKind
from uacpy.output import logging

if TYPE_CHECKING:
    import pyvista as pv


# ==================================================================================================
@dataclass(eq=False)
class PathNode:
    """A resolved `BoundaryPath`/`ConnectionPath`: 1-cell, holds the raw path once constructed.

    Attributes:
        key: Key sequence of this path in the config.
        config: The unresolved config.
        start: Node of the connection's start path or marker. `None` for a `BoundaryPath`.
        end: Node of the connection's end path or marker. `None` for a `BoundaryPath`.
        inadmissible_contact: Paths the connection may not touch.
        inadmissible_along: Paths the connection may not travel along.
        result: Ordered vertex indices into the original mesh, once constructed.
    """

    key: dict_utils.KeySequence
    config: model.BoundaryPath | model.ConnectionPath
    start: PathNode | MarkerNode | None = None
    end: PathNode | MarkerNode | None = None
    inadmissible_contact: list[PathNode] = field(default_factory=list)
    inadmissible_along: list[PathNode] = field(default_factory=list)
    result: np.ndarray | None = None

    @property
    def dependencies(self) -> list[PathNode | MarkerNode]:
        """The other nodes this path's own construction reads from.

        Returns:
            The start and end nodes (if any) plus all inadmissible-path nodes.
        """
        dependencies = [node for node in (self.start, self.end) if node is not None]
        dependencies.extend(self.inadmissible_contact)
        dependencies.extend(self.inadmissible_along)
        return dependencies

    def construct(
        self, mesh: pv.PolyData, feature_tags: dict[str, int], logger: logging.BaseLogger
    ) -> None:
        """Extract a `BoundaryPath` by feature tag, or a `ConnectionPath` as a shortest path.

        Args:
            mesh: Triangle surface mesh with `anatomical_tags` cell data.
            feature_tags: Map from tag names to the integer values in `anatomical_tags`.
            logger: Receives debug messages.

        Raises:
            RuntimeError: If a dependency has not been constructed yet.
            ValueError: If a connection's boundary type is unknown.
        """
        if isinstance(self.config, model.BoundaryPath):
            tag_value = feature_tags[self.config.feature_tag]
            is_mesh_boundary = self.config.coincides_with_mesh_boundary
            logger.debug(
                f"{logging.format_key(self.key)}: feature '{self.config.feature_tag}' = tag "
                f"{tag_value}, coincides_with_mesh_boundary={is_mesh_boundary}"
            )
            self.result = geometry.get_feature_boundary(mesh, tag_value, is_mesh_boundary, logger)
            logger.debug(f"  -> boundary of {len(self.result)} vertices")
            return

        boundaries = self._resolve_connection_boundaries()
        inadmissible_sets = self._resolve_inadmissible_sets()
        logger.debug(
            f"{logging.format_key(self.key)}: shortest path between sets of "
            f"{len(boundaries[0])} and {len(boundaries[1])} vertices, inadmissible "
            f"contact/along sets of {len(inadmissible_sets[0])}/{len(inadmissible_sets[1])} "
            "vertices"
        )
        self.result = geometry.construct_shortest_path_between_subsets(
            mesh, *boundaries, *inadmissible_sets, logger=logger
        )
        logger.debug(f"  -> path of {len(self.result)} vertices")

    def _resolve_connection_boundaries(self) -> list[np.ndarray]:
        """Resolve `start` and `end` to vertex-index sets (a raw path or a single marker)."""
        boundaries = []
        for boundary_type, boundary_node in zip(
            self.config.boundary_types, (self.start, self.end), strict=True
        ):
            if boundary_type == EndpointKind.PATH:
                boundaries.append(_require(boundary_node, self))
            elif boundary_type == EndpointKind.MARKER:
                boundaries.append(np.array((_require(boundary_node, self),), dtype=int))
            else:
                raise ValueError(
                    f"Unknown boundary type {boundary_type} for path {logging.format_key(self.key)}"
                )
        return boundaries

    def _resolve_inadmissible_sets(self) -> list[np.ndarray]:
        """Resolve the inadmissible-path nodes to vertex-index sets."""
        inadmissible_sets = []
        for path_nodes in (self.inadmissible_contact, self.inadmissible_along):
            if not path_nodes:
                inadmissible_sets.append(np.array([], dtype=int))
            else:
                subsets = [_require(path_node, self) for path_node in path_nodes]
                inadmissible_sets.append(np.unique(np.concatenate(subsets)))
        return inadmissible_sets


# ==================================================================================================
@dataclass(eq=False)
class MarkerNode:
    """A resolved `Marker`: 0-cell, holds its vertex index once extracted.

    Attributes:
        key: Key sequence of this marker in the config.
        config: The unresolved config.
        source: Node of the path (for index positions) or parameterization (for relative
            positions) the marker is read off.
        result: Vertex index into the original mesh, once constructed.
    """

    key: dict_utils.KeySequence
    config: model.Marker
    source: PathNode | ParameterizationNode | None = None
    result: int | None = None

    @property
    def dependencies(self) -> list[PathNode | ParameterizationNode]:
        """The other node this marker's own position is read off of.

        Returns:
            A list with the source node, or an empty list if unresolved.
        """
        return [self.source] if self.source is not None else []

    def construct(self, logger: logging.BaseLogger) -> None:
        """Resolve a `Marker` to a vertex index, by raw-path index or relative arclength.

        For relative positions, the first path vertex at or beyond the prescribed relative length
        is chosen.

        Args:
            logger: Receives debug messages.

        Raises:
            RuntimeError: If the source has not been constructed yet.
            IndexError: If an index position is out of bounds for the path.
            ValueError: If a relative position is not on the path, or the position type is
                unknown.
        """

        if self.config.position_type == PositionKind.INDEX:
            containing_path = _require(self.source, self)
            logger.debug(
                f"{logging.format_key(self.key)}: index {self.config.position} on "
                f"{logging.format_key(self.source.key)} ({len(containing_path)} vertices)"
            )
            try:
                marker_ind = containing_path[self.config.position]
            except IndexError as e:
                raise IndexError(
                    f"Index {self.config.position} out of bounds for path "
                    f"{logging.format_key(self.source.key)} with length "
                    f"{len(containing_path)}"
                ) from e

        elif self.config.position_type == PositionKind.RELATIVE:
            containing_path = _require(self.source, self)
            logger.debug(
                f"{logging.format_key(self.key)}: relative length {self.config.position} on "
                f"{logging.format_key(self.source.key)} ({len(containing_path.inds)} vertices)"
            )
            try:
                relative_marker_ind = np.where(
                    containing_path.relative_lengths >= self.config.position
                )[0][0]
                marker_ind = containing_path.inds[relative_marker_ind]
            except IndexError as e:
                raise ValueError(
                    f"Relative position {self.config.position} not found on path "
                    f"{logging.format_key(self.source.key)}"
                ) from e
        else:
            raise ValueError(
                f"Unknown position_type {self.config.position_type} for marker "
                f"{logging.format_key(self.key)}"
            )

        self.result = int(marker_ind)
        logger.debug(f"  -> vertex {self.result}")


# ==================================================================================================
@dataclass(eq=False)
class ParameterizationNode:
    """A resolved `PathParameterization`, holds the arclength parameterization once computed.

    Attributes:
        key: Key sequence of this parameterization in the config.
        config: The unresolved config.
        path: Node of the raw path being parameterized.
        markers: Nodes of the markers pinned along the path.
        result: The arclength-parameterized path, once constructed.
    """

    key: dict_utils.KeySequence
    config: model.PathParameterization
    path: PathNode | None = None
    markers: list[MarkerNode] = field(default_factory=list)
    result: model.ParameterizedPath | None = None

    @property
    def dependencies(self) -> list[PathNode | MarkerNode]:
        """The other nodes this parameterization's own construction reads from.

        Returns:
            The path node (first, if resolved) followed by the marker nodes.
        """
        dependencies: list[PathNode | MarkerNode] = list(self.markers)
        if self.path is not None:
            dependencies.insert(0, self.path)
        return dependencies

    def construct(self, mesh: pv.PolyData, logger: logging.BaseLogger) -> None:
        """Arclength-parameterize a raw path between its markers.

        Args:
            mesh: Triangle surface mesh providing vertex positions.
            logger: Receives debug messages.

        Raises:
            RuntimeError: If the path or a marker has not been constructed yet.
        """
        path = _require(self.path, self)
        marker_inds = [_require(marker_node, self) for marker_node in self.markers]
        self.result = geometry.parameterize_path(
            mesh, path, marker_inds, self.config.marker_relative_positions
        )
        arclength = np.linalg.norm(np.diff(mesh.points[self.result.inds], axis=0), axis=1).sum()
        logger.debug(
            f"{logging.format_key(self.key)}: {len(self.result.inds)} vertices, arclength "
            f"{arclength:.4g}, markers {marker_inds} at relative lengths "
            f"{list(self.config.marker_relative_positions)}"
        )


# ==================================================================================================
@dataclass(eq=False)
class UACMappingNode:
    """A resolved `UACMapping`, holds the UAC-mapped path once computed.

    Attributes:
        key: Key sequence of this mapping in the config.
        config: The unresolved config.
        parameterization: Node of the parameterization being mapped.
        result: The path with (alpha, beta) at each vertex, once constructed.
    """

    key: dict_utils.KeySequence
    config: model.UACMapping
    parameterization: ParameterizationNode | None = None
    result: model.UACPath | None = None

    def construct(self, logger: logging.BaseLogger) -> None:
        """Assign target (alpha, beta) coordinates along the parameterization's arclength.

        Args:
            logger: Receives debug messages.

        Raises:
            RuntimeError: If the parameterization has not been constructed yet.
        """
        parameterized_path = _require(self.parameterization, self)
        self.result = geometry.compute_uacs_polyline(
            parameterized_path, self.config.relative_positions, self.config.uacs
        )
        logger.debug(
            f"{logging.format_key(self.key)}: {len(self.result.inds)} vertices, "
            f"alpha {logging.format_range(self.result.alpha)}, "
            f"beta {logging.format_range(self.result.beta)}"
        )


# ==================================================================================================
@dataclass(eq=False)
class SegmentNode:
    """A resolved `Segment`: 2-cell, holds its boundary/submesh/UAC results once computed.

    Attributes:
        key: Key sequence of this segment in the config.
        config: The unresolved config.
        boundary_paths: Nodes of the UAC-mapped paths forming the segment's boundary.
        outside_path: Node of a raw path outside the segment.
        boundary_result: The segment's closed, ordered boundary loop with (alpha, beta).
        submesh_result: The segment's region of the original mesh.
        uac_submesh_result: The region with (alpha, beta) at every vertex.
    """

    key: dict_utils.KeySequence
    config: model.Segment
    boundary_paths: list[UACMappingNode] = field(default_factory=list)
    outside_path: PathNode | None = None
    boundary_result: model.UACPath | None = None
    submesh_result: model.Submesh | None = None
    uac_submesh_result: model.UACSubmesh | None = None

    def construct_boundary(self, logger: logging.BaseLogger) -> None:
        """Stitch this segment's boundary portions into one closed, ordered loop.

        Each `boundary_paths` entry contributes only the sub-range of its already-UAC-mapped path
        that `portions` (a `(start, end)` pair in relative arclength) says belongs to this segment
        -- so two segments sharing a path each get a disjoint slice of it, not the whole thing.
        `_concatenate_submesh_boundaries` then walks those slices end-to-end into a single loop.

        Args:
            logger: Receives debug messages.

        Raises:
            RuntimeError: If a boundary path has not been constructed yet.
            ValueError: If the slices do not form a closed loop.
        """
        boundary_inds, boundary_alpha, boundary_beta = [], [], []
        for uac_node, portion in zip(self.boundary_paths, self.config.portions, strict=True):
            uac_path = _require(uac_node, self)
            relevant_section = np.where(
                (uac_path.relative_lengths >= portion[0])
                & (uac_path.relative_lengths <= portion[1])
            )[0]
            logger.debug(
                f"{logging.format_key(self.key)}: portion {tuple(portion)} of "
                f"{logging.format_key(uac_node.key)} takes {relevant_section.size} of "
                f"{len(uac_path.inds)} vertices"
            )
            boundary_inds.append(uac_path.inds[relevant_section])
            boundary_alpha.append(uac_path.alpha[relevant_section])
            boundary_beta.append(uac_path.beta[relevant_section])

        num_vertices_before = sum(len(inds) for inds in boundary_inds)
        unique_inds, unique_alpha, unique_beta = self._concatenate_submesh_boundaries(
            boundary_inds, boundary_alpha, boundary_beta
        )
        self.boundary_result = model.UACPath(inds=unique_inds, alpha=unique_alpha, beta=unique_beta)
        logger.debug(
            f"  -> loop of {len(unique_inds)} vertices from "
            f"{num_vertices_before} before removing duplicates"
        )

    @staticmethod
    def _concatenate_submesh_boundaries(
        boundary_inds: list[np.ndarray],
        boundary_alpha: list[np.ndarray],
        boundary_beta: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Walk unordered boundary-portion slices end-to-end into one closed loop.

        Each `boundary_*[i]` is one portion's vertex sequence, in its own path's original
        direction -- not necessarily the direction that continues the loop, and not necessarily
        adjacent to `boundary_*[i + 1]` in the input order. Each step finds whichever remaining
        portion shares its endpoint with the loop built so far, flips it if it runs the wrong way,
        and appends it; closing the loop (last endpoint == first) drops the duplicated closing
        vertex. The input lists are consumed.
        """
        unique_inds = boundary_inds.pop(0)
        unique_alpha = boundary_alpha.pop(0)
        unique_beta = boundary_beta.pop(0)

        current_end_point = unique_inds[-1]
        while boundary_inds:
            for i, inds in enumerate(boundary_inds):
                if current_end_point in inds:
                    next_segment_inds = boundary_inds.pop(i)
                    next_segment_alpha = boundary_alpha.pop(i)
                    next_segment_beta = boundary_beta.pop(i)
                    break
            else:
                raise ValueError("Submesh boundary segments do not form a closed loop.")
            if current_end_point == next_segment_inds[-1]:
                next_segment_inds = next_segment_inds[::-1]
                next_segment_alpha = next_segment_alpha[::-1]
                next_segment_beta = next_segment_beta[::-1]
            unique_inds = np.append(unique_inds, next_segment_inds[1:])
            unique_alpha = np.append(unique_alpha, next_segment_alpha[1:])
            unique_beta = np.append(unique_beta, next_segment_beta[1:])
            current_end_point = unique_inds[-1]

        if unique_inds[0] == unique_inds[-1]:
            unique_inds = unique_inds[:-1]
            unique_alpha = unique_alpha[:-1]
            unique_beta = unique_beta[:-1]
        return unique_inds, unique_alpha, unique_beta

    def construct_submesh(self, mesh: pv.PolyData, logger: logging.BaseLogger) -> None:
        """Flood-fill the original mesh from this segment's boundary loop to its interior.

        Args:
            mesh: Triangle surface mesh the boundary indices refer to.
            logger: Receives debug messages.

        Raises:
            RuntimeError: If the boundary or the outside path has not been constructed yet.
        """
        if self.boundary_result is None:
            raise RuntimeError(
                f"Submesh boundary {logging.format_key(self.key)} has not been constructed yet."
            )
        outside_path = _require(self.outside_path, self)
        logger.debug(
            f"{logging.format_key(self.key)}: flood fill inside a boundary of "
            f"{len(self.boundary_result.inds)} vertices"
        )
        self.submesh_result = geometry.extract_region_from_boundary(
            mesh, self.boundary_result.inds, outside_path, logger
        )
        logger.debug(
            f"  -> {len(self.submesh_result.vertex_inds)} vertices and "
            f"{len(self.submesh_result.cell_inds)} cells "
            f"({len(self.submesh_result.cell_inds) / mesh.n_cells:.1%} of the mesh)"
        )

    def construct_uacs(self, mesh: pv.PolyData, logger: logging.BaseLogger) -> None:
        """Extend the boundary's (alpha, beta) values over the submesh interior via `mvc`.

        Mean-value coordinates reproduce the boundary values exactly and interpolate smoothly
        inside, so `boundary_result`'s already-assigned coordinates become boundary conditions
        for the interior vertices' own (alpha, beta).

        Args:
            mesh: Triangle surface mesh providing vertex positions.
            logger: Receives debug messages.

        Raises:
            RuntimeError: If the boundary or submesh has not been constructed yet.
        """
        submesh_boundary = self.boundary_result
        submesh = self.submesh_result
        if submesh_boundary is None or submesh is None:
            raise RuntimeError(
                f"Submesh {logging.format_key(self.key)} has not been extracted yet."
            )

        vertices = np.array(mesh.points[submesh.vertex_inds])
        simplices = submesh.connectivity
        boundary_inds = np.array(
            [np.where(submesh.vertex_inds == ind)[0][0] for ind in submesh_boundary.inds]
        )
        uac_coordinates = np.hstack(
            (submesh_boundary.alpha[:, None], submesh_boundary.beta[:, None])
        )
        mvc_start = time.perf_counter()
        mvc_map = mvc.compute_mean_value_coordinates(
            vertices, simplices, boundary_inds, uac_coordinates
        )
        logger.debug(
            f"{logging.format_key(self.key)}: {len(boundary_inds)} boundary and "
            f"{len(vertices) - len(boundary_inds)} interior vertices, "
            f"alpha {logging.format_range(mvc_map[:, 0])}, "
            f"beta {logging.format_range(mvc_map[:, 1])}, "
            f"MVC took {time.perf_counter() - mvc_start:.3f} s"
        )
        self.uac_submesh_result = model.UACSubmesh(
            vertex_inds=submesh.vertex_inds,
            connectivity=simplices,
            cell_inds=submesh.cell_inds,
            alpha=mvc_map[:, 0],
            beta=mvc_map[:, 1],
        )


type Node = PathNode | MarkerNode | ParameterizationNode | UACMappingNode | SegmentNode


# ==================================================================================================
@dataclass
class ConstructionGraph:
    """The whole config, resolved: one dict of nodes per layer, keyed by key-sequence.

    Attributes:
        paths: Path nodes.
        markers: Marker nodes.
        parameterizations: Parameterization nodes.
        uacs: UAC mapping nodes.
        segments: Segment nodes.
    """

    paths: dict[dict_utils.KeySequence, PathNode]
    markers: dict[dict_utils.KeySequence, MarkerNode]
    parameterizations: dict[dict_utils.KeySequence, ParameterizationNode]
    uacs: dict[dict_utils.KeySequence, UACMappingNode]
    segments: dict[dict_utils.KeySequence, SegmentNode]


# ==================================================================================================
def link(
    path_config: dict,
    marker_config: dict,
    parameterization_config: dict,
    uac_config: dict,
    submesh_config: dict,
) -> ConstructionGraph:
    """Resolve five key-sequence-addressed config dicts into a `ConstructionGraph` of live nodes.

    Two passes: first every entry becomes a node (so any entry can be referenced regardless of
    which order the configs were authored in), then each node's key-sequence-valued fields are
    resolved into direct references to the nodes they name.

    Args:
        path_config: Nested dict of `BoundaryPath` and `ConnectionPath`.
        marker_config: Nested dict of `Marker`.
        parameterization_config: Nested dict of `PathParameterization`.
        uac_config: Nested dict of `UACMapping`.
        submesh_config: Nested dict of `Segment`.

    Returns:
        The linked graph, with all results still unset.

    Raises:
        KeyError: If any key-sequence reference names an entry that does not exist.
    """
    paths = {
        key: PathNode(key=key, config=config)
        for key, config in dict_utils.flatten(path_config).items()
    }
    markers = {
        key: MarkerNode(key=key, config=config)
        for key, config in dict_utils.flatten(marker_config).items()
    }
    parameterizations = {
        key: ParameterizationNode(key=key, config=config)
        for key, config in dict_utils.flatten(parameterization_config).items()
    }
    uacs = {
        key: UACMappingNode(key=key, config=config)
        for key, config in dict_utils.flatten(uac_config).items()
    }
    segments = {
        key: SegmentNode(key=key, config=config)
        for key, config in dict_utils.flatten(submesh_config).items()
    }

    for node in paths.values():
        if isinstance(node.config, model.BoundaryPath):
            continue
        start_type, end_type = node.config.boundary_types
        node.start = _resolve(
            paths if start_type == EndpointKind.PATH else markers, node.config.start
        )
        node.end = _resolve(paths if end_type == EndpointKind.PATH else markers, node.config.end)
        node.inadmissible_contact = [
            _resolve(paths, key) for key in node.config.inadmissible_contact or ()
        ]
        node.inadmissible_along = [
            _resolve(paths, key) for key in node.config.inadmissible_along or ()
        ]

    for node in markers.values():
        source_table = (
            paths if node.config.position_type == PositionKind.INDEX else parameterizations
        )
        node.source = _resolve(source_table, node.config.path)

    for node in parameterizations.values():
        node.path = _resolve(paths, node.config.path)
        node.markers = [_resolve(markers, key) for key in node.config.markers]

    for node in uacs.values():
        node.parameterization = _resolve(parameterizations, node.config.path)

    for node in segments.values():
        node.boundary_paths = [_resolve(uacs, key) for key in node.config.boundary_paths]
        node.outside_path = _resolve(paths, node.config.outside_path)

    return ConstructionGraph(
        paths=paths,
        markers=markers,
        parameterizations=parameterizations,
        uacs=uacs,
        segments=segments,
    )


# --------------------------------------------------------------------------------------------------
def _resolve[T](table: dict[dict_utils.KeySequence, T], key_sequence: dict_utils.KeySequence) -> T:
    """Look up the node a key sequence names."""
    key = tuple(key_sequence)
    try:
        return table[key]
    except KeyError as e:
        raise KeyError(
            f"Dangling reference {logging.format_key(key)} not found while linking the config"
        ) from e


# --------------------------------------------------------------------------------------------------
def _require(node: Node | None, dependent: Node) -> object:
    """Get the result a node has already computed."""
    if node is None or node.result is None:
        key = logging.format_key(node.key) if node else "?"
        raise RuntimeError(
            f"{key} has not been constructed yet (required by {logging.format_key(dependent.key)})."
        )
    return node.result

"""`UACConstructor`, the thin orchestrator that runs a linked `graph.ConstructionGraph`."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pyvista as pv

from uacpy.common import dict_utils
from uacpy.construction import graph, model
from uacpy.output import logging

# ==================================================================================================
_STEP_COLUMNS = ("Step", "Kind", "Key", "Time [s]")
_STEP_ALIGNMENTS = "><<>"
_TABLE_RULE_CHARACTER = "-"


# ==================================================================================================
@dataclass
class UACConstructorSettings:
    """Everything `UACConstructor` needs: the mesh, its feature tags, and the five config dicts.

    `feature_tags` maps each `model.BoundaryPath.feature_tag` string to the integer value
    actually stored in the mesh's `"anatomical_tags"` cell data.

    Attributes:
        mesh: Triangle surface mesh with `anatomical_tags` cell data.
        feature_tags: Map from tag names to values in `anatomical_tags`.
        path_config: Nested dict of `BoundaryPath` and `ConnectionPath`.
        marker_config: Nested dict of `Marker`.
        parameterization_config: Nested dict of `PathParameterization`.
        uac_config: Nested dict of `UACMapping`.
        submesh_config: Nested dict of `Segment`.
        logger: Receives the progress log. The caller owns it and closes it. If `None`, a
            console-only logger is created.

    The config dicts are nested as authored; `graph.link` flattens them.
    """

    mesh: pv.PolyData
    feature_tags: dict[str, int]
    path_config: dict
    marker_config: dict
    parameterization_config: dict
    uac_config: dict
    submesh_config: dict
    logger: logging.BaseLogger | None = None


@dataclass
class UACData:
    """A snapshot of `UACConstructor`'s results, as flat dicts keyed by full key sequence.

    This is the pipeline's external data contract -- `check.py`, `output/`, `mapper.py` and the
    notebooks all consume it in this shape. Every dict is keyed by the tuple key sequence of the
    config entry that produced it, e.g. `("LIPV", "segment_1")`. Internally, `UACConstructor`
    holds a `graph.ConstructionGraph` instead. Values are `None` for results not computed yet.

    Attributes:
        marker_data: Marker vertex indices.
        raw_path_data: Ordered vertex indices of each raw path.
        parameterized_path_data: Arclength-parameterized paths.
        uac_path_data: Paths with (alpha, beta) at each vertex.
        submesh_boundary_data: Closed boundary loop of each segment, with (alpha, beta).
        submesh_data: Each segment's region of the original mesh.
        uac_submesh_data: Each segment's region with (alpha, beta) at every vertex.
    """

    marker_data: dict_utils.KeySequenceDict[int]
    raw_path_data: dict_utils.KeySequenceDict[np.ndarray]
    parameterized_path_data: dict_utils.KeySequenceDict[model.ParameterizedPath]
    uac_path_data: dict_utils.KeySequenceDict[model.UACPath]
    submesh_boundary_data: dict_utils.KeySequenceDict[model.UACPath]
    submesh_data: dict_utils.KeySequenceDict[model.Submesh]
    uac_submesh_data: dict_utils.KeySequenceDict[model.UACSubmesh]


class UACConstructor:
    """Orchestrates the construction pipeline over a linked `graph.ConstructionGraph`.

    This class only decides *what order* to do work in (topological sort for segmentation, then
    the fixed UAC/submesh/MVC phases); it doesn't implement *how* any individual node is computed
    -- that lives on the node types themselves in `graph.py`. Keeping those separate is what keeps
    this class from becoming a god object that both orchestrates and does everything itself.

    Call `construct_segmentation`, then `construct_uacs`, then read `data`.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: UACConstructorSettings) -> None:
        """Link the configs into a construction graph; no mesh work happens yet.

        Args:
            settings: The mesh, feature tags, and config dicts.

        Raises:
            KeyError: If a config contains a dangling key-sequence reference.
        """
        self._mesh = settings.mesh
        self._feature_tags = settings.feature_tags
        self._logger = logging.resolve_logger(settings.logger, print_to_console=True)
        self._graph = graph.link(
            path_config=settings.path_config,
            marker_config=settings.marker_config,
            parameterization_config=settings.parameterization_config,
            uac_config=settings.uac_config,
            submesh_config=settings.submesh_config,
        )
        self._logger.debug(
            f"Linked construction graph: {len(self._graph.paths)} paths, "
            f"{len(self._graph.markers)} markers, {len(self._graph.parameterizations)} "
            f"parameterizations, {len(self._graph.uacs)} UAC mappings, "
            f"{len(self._graph.segments)} segments; mesh with {self._mesh.n_points} vertices and "
            f"{self._mesh.n_cells} cells; feature tags {self._feature_tags}"
        )

    # ----------------------------------------------------------------------------------------------
    def construct_segmentation(self) -> None:
        """Extract features/markers/paths/parameterizations in dependency order.

        These three layers feed each other back and forth rather than in fixed phases -- a marker
        split off one connector can feed a second connector, whose own markers feed a further
        parameterization -- so the order isn't a hand-sequenced pipeline of stages. Instead, each
        node's own `dependencies` (resolved by `graph.link`) say what it reads from, and a
        `networkx` topological sort over that graph gives a valid execution order.

        Raises:
            ValueError: If the config's dependencies are circular.
        """
        dependency_graph = self._build_segmentation_graph()
        try:
            order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible as e:
            cycle = nx.find_cycle(dependency_graph)
            raise ValueError(f"Segmentation config has a circular dependency: {cycle}") from e

        for i, node in enumerate(order, start=1):
            self._logger.debug(
                f"Order {i}: {logging.format_key(node.key)} depends on "
                f"{len(node.dependencies)} node(s)"
            )
        steps = [
            (
                self._segmentation_kind(node),
                logging.format_key(node.key),
                lambda node=node: self._construct_segmentation_node(node),
            )
            for node in order
        ]
        self._run_section("Segmentation", steps)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _segmentation_kind(
        node: graph.PathNode | graph.MarkerNode | graph.ParameterizationNode,
    ) -> str:
        """Short label of what a segmentation node builds, for the progress table."""
        match node:
            case graph.PathNode(config=model.BoundaryPath()):
                return "feature"
            case graph.PathNode():
                return "shortest path"
            case graph.MarkerNode():
                return "marker"
            case graph.ParameterizationNode():
                return "parameterization"

    # ----------------------------------------------------------------------------------------------
    def _construct_segmentation_node(
        self, node: graph.PathNode | graph.MarkerNode | graph.ParameterizationNode
    ) -> None:
        """Construct one path, marker, or parameterization node."""
        match node:
            case graph.PathNode():
                node.construct(self._mesh, self._feature_tags, self._logger)
            case graph.MarkerNode():
                node.construct(self._logger)
            case graph.ParameterizationNode():
                node.construct(self._mesh, self._logger)

    # ----------------------------------------------------------------------------------------------
    def _run_section(self, title: str, steps: list[tuple[str, str, Callable[[], None]]]) -> None:
        """Run the steps of one pipeline section, logging a divider and a timed table of steps.

        Args:
            title: Name of the section.
            steps: One `(kind, key, action)` per step: a short label of what is built, the
                formatted key of the entry it is built for, and the call that builds it.
        """
        step_labels = [f"{i}/{len(steps)}" for i in range(1, len(steps) + 1)]
        columns = [step_labels, [kind for kind, _, _ in steps], [key for _, key, _ in steps]]
        widths = [
            max([len(header), *(len(cell) for cell in column)])
            for header, column in zip(_STEP_COLUMNS, columns, strict=False)
        ]
        widths.append(len(_STEP_COLUMNS[-1]))
        rule = logging.format_table_row(
            [_TABLE_RULE_CHARACTER * width for width in widths], widths, _STEP_ALIGNMENTS
        )

        self._logger.section(title)
        self._logger.debug(f"Section '{title}' started with {len(steps)} steps")
        self._logger.info(logging.format_table_row(_STEP_COLUMNS, widths, _STEP_ALIGNMENTS))
        self._logger.info(rule)
        section_start = time.perf_counter()
        step_durations = []
        for label, (kind, key, action) in zip(step_labels, steps, strict=True):
            step_start = time.perf_counter()
            action()
            step_durations.append(time.perf_counter() - step_start)
            row = (label, kind, key, f"{step_durations[-1]:.3f}")
            self._logger.info(logging.format_table_row(row, widths, _STEP_ALIGNMENTS))
        self._logger.info(rule)
        self._logger.info(f"{len(steps)} steps in {time.perf_counter() - section_start:.2f} s")
        if steps:
            slowest = int(np.argmax(step_durations))
            self._logger.debug(
                f"Section '{title}' slowest step: {steps[slowest][1]} "
                f"({step_durations[slowest]:.3f} s)"
            )

    # ----------------------------------------------------------------------------------------------
    def _build_segmentation_graph(self) -> nx.DiGraph:
        """Build the dependency graph over path, marker, and parameterization nodes."""
        nodes = [
            *self._graph.paths.values(),
            *self._graph.markers.values(),
            *self._graph.parameterizations.values(),
        ]
        dependency_graph = nx.DiGraph()
        dependency_graph.add_nodes_from(nodes)
        for node in nodes:
            for dependency in node.dependencies:
                dependency_graph.add_edge(dependency, node)
        return dependency_graph

    # ----------------------------------------------------------------------------------------------
    def construct_uacs(self) -> None:
        """Run the UAC/submesh phases, each a fixed pass over every segment in turn.

        Unlike `construct_segmentation`, this order isn't derived -- assigning UAC targets,
        stitching boundaries, flood-filling submeshes, and interpolating (alpha, beta) over each
        interior are always in this order for every segment, so a plain fixed sequence of phases
        is correct and there's no dependency graph to sort. Requires `construct_segmentation` to
        have run first.

        Raises:
            RuntimeError: If a node's dependency has not been constructed yet.
        """
        uac_nodes = list(self._graph.uacs.values())
        segment_nodes = list(self._graph.segments.values())
        self._run_section(
            "UAC paths",
            [
                ("uac path", logging.format_key(n.key), lambda n=n: n.construct(self._logger))
                for n in uac_nodes
            ],
        )
        self._run_section(
            "Submesh boundaries",
            [
                (
                    "boundary",
                    logging.format_key(n.key),
                    lambda n=n: n.construct_boundary(self._logger),
                )
                for n in segment_nodes
            ],
        )
        self._run_section(
            "Submeshes",
            [
                (
                    "submesh",
                    logging.format_key(n.key),
                    lambda n=n: n.construct_submesh(self._mesh, self._logger),
                )
                for n in segment_nodes
            ],
        )
        self._run_section(
            "UACs on submeshes",
            [
                (
                    "mvc",
                    logging.format_key(n.key),
                    lambda n=n: n.construct_uacs(self._mesh, self._logger),
                )
                for n in segment_nodes
            ],
        )

    # ----------------------------------------------------------------------------------------------
    @property
    def data(self) -> UACData:
        """Collect the graph's results into the flat keyed dicts of the external data contract.

        Returns:
            A snapshot of all results computed so far, rebuilt on every access.
        """
        return UACData(
            marker_data={node.key: node.result for node in self._graph.markers.values()},
            raw_path_data={node.key: node.result for node in self._graph.paths.values()},
            parameterized_path_data={
                node.key: node.result for node in self._graph.parameterizations.values()
            },
            uac_path_data={node.key: node.result for node in self._graph.uacs.values()},
            submesh_boundary_data={
                node.key: node.boundary_result for node in self._graph.segments.values()
            },
            submesh_data={node.key: node.submesh_result for node in self._graph.segments.values()},
            uac_submesh_data={
                node.key: node.uac_submesh_result for node in self._graph.segments.values()
            },
        )

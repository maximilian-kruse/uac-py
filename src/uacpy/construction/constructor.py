"""`UACConstructor`, the thin orchestrator that runs a linked `graph.ConstructionGraph`."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pyvista as pv

from uacpy.common import dict_utils
from uacpy.construction import graph, model
from uacpy.output import logging


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

    The config dicts are nested as authored; `graph.link` flattens them.
    """

    mesh: pv.PolyData
    feature_tags: dict[str, int]
    path_config: dict
    marker_config: dict
    parameterization_config: dict
    uac_config: dict
    submesh_config: dict


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
        self._graph = graph.link(
            path_config=settings.path_config,
            marker_config=settings.marker_config,
            parameterization_config=settings.parameterization_config,
            uac_config=settings.uac_config,
            submesh_config=settings.submesh_config,
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
        print("Starting Segmentation")
        print("=====================\n\n")

        dependency_graph = self._build_segmentation_graph()
        try:
            order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXUnfeasible as e:
            cycle = nx.find_cycle(dependency_graph)
            raise ValueError(f"Segmentation config has a circular dependency: {cycle}") from e

        for i, node in enumerate(order):
            step_key = logging.format_key(node.key)
            logging.print_with_underline(f"Step {i + 1}/{len(order)}: {step_key}")
            match node:
                case graph.PathNode():
                    node.construct(self._mesh, self._feature_tags)
                case graph.MarkerNode():
                    node.construct()
                case graph.ParameterizationNode():
                    node.construct(self._mesh)
            print(" ")

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
        logging.print_with_underline("Construct UAC Paths")
        for node in self._graph.uacs.values():
            node.construct()
        print(" ")
        logging.print_with_underline("Extract Submesh Boundaries")
        for node in self._graph.segments.values():
            node.construct_boundary()
        print(" ")
        logging.print_with_underline("Extract Submeshes")
        for node in self._graph.segments.values():
            node.construct_submesh(self._mesh)
        print(" ")
        logging.print_with_underline("Compute UACs on Submeshes")
        for node in self._graph.segments.values():
            node.construct_uacs(self._mesh)
        print(" ")

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

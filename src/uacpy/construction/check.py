"""Validity checks for the construction pipeline, each catching a defect the others can't.

`check_uac_mesh_validity` is geometric, on the computed submeshes, and needs a full pipeline run.
`check_segment_boundary_multiplicity` is structural, on the config alone, and needs no mesh --
it's the one to run first, since it catches a config mistake before spending time on a pipeline
run at all. `check_uac_tiling` is geometric again, but on the target UAC atlas rather than the
source mesh, so it needs the pipeline's results but not a submesh triangulation.
"""

import numpy as np
import shapely

from uacpy.common import dict_utils
from uacpy.construction import graph, model
from uacpy.output import logging

# ==================================================================================================
_TILING_AREA_TOLERANCE = 1e-9


# ==================================================================================================
def check_uac_mesh_validity(uac_submesh: dict_utils.KeySequenceDict[model.UACSubmesh]) -> bool:
    """No submesh should fold over itself in UAC space (a cell with non-positive area).

    A fold-over means the mean-value-coordinate map placed two or more original-mesh cells on top
    of each other in (alpha, beta) space -- always a defect, never a valid configuration, so unlike
    `check_uac_tiling` there's no legitimate case to carve out here.

    Args:
        uac_submesh: `UACSubmesh` per segment (`UACData.uac_submesh_data`).

    Returns:
        `True` if every cell of every submesh has positive area. Otherwise prints the first
        offending submesh and returns `False`.
    """
    for key, submesh_data in uac_submesh.items():
        pv_submesh = submesh_data.to_polydata()
        quality_data = pv_submesh.cell_quality(quality_measure="area")
        cell_areas = quality_data.cell_data["area"]

        if np.any(cell_areas <= 0):
            formatted_key = logging.format_key(key)
            print(
                f"Foldover detected in submesh {formatted_key}: found "
                f"{np.sum(cell_areas <= 0)} cells with non-positive area."
            )
            return False

    return True


# ==================================================================================================
def check_segment_boundary_multiplicity(
    path_config: dict,
    marker_config: dict,
    parameterization_config: dict,
    uac_config: dict,
    submesh_config: dict,
) -> bool:
    """Every Segment-boundary portion should be used by 1 (mesh edge) or 2 (shared) segments.

    A boundary path's *portion* is the unit here, not the path itself: a PV ostium's inner loop is
    one path but several distinct portions of it, each its own edge shared with a different
    neighbour (e.g. one arc borders that PV's own segment_1 and "roof"; a different, non-
    overlapping arc borders segment_2 and "lateral_roof"). Grouping by (path, portion) rather than
    by path alone is what tells those apart -- checked directly on the config, before any mesh
    work, so a config mistake (a forgotten or duplicated boundary reference) is caught immediately
    rather than surfacing later as a geometric defect in `check_uac_tiling`.

    Args:
        path_config: Nested dict of path configs.
        marker_config: Nested dict of marker configs.
        parameterization_config: Nested dict of path parameterization configs.
        uac_config: Nested dict of UAC mapping configs.
        submesh_config: Nested dict of segment configs.

    Returns:
        `True` if every boundary portion is referenced by one or two segments. Otherwise prints
        each offending portion and returns `False`.

    Raises:
        KeyError: If the config contains a dangling key-sequence reference.
    """
    construction_graph = graph.link(
        path_config=path_config,
        marker_config=marker_config,
        parameterization_config=parameterization_config,
        uac_config=uac_config,
        submesh_config=submesh_config,
    )

    segments_by_boundary_portion: dict[tuple, list[tuple]] = {}
    for segment in construction_graph.segments.values():
        for uac_node, portion in zip(segment.boundary_paths, segment.config.portions, strict=True):
            rounded_portion = tuple(round(bound, 9) for bound in portion)
            key = (uac_node.key, rounded_portion)
            segments_by_boundary_portion.setdefault(key, []).append(segment.key)

    is_valid = True
    for (boundary_key, portion), segment_keys in segments_by_boundary_portion.items():
        if len(segment_keys) not in (1, 2):
            formatted_segment_keys = [logging.format_key(key) for key in segment_keys]
            print(
                f"Boundary {logging.format_key(boundary_key)} portion {portion} is "
                f"referenced by {len(segment_keys)} segments {formatted_segment_keys} "
                "-- expected 1 or 2."
            )
            is_valid = False

    return is_valid


# ==================================================================================================
def check_uac_tiling(submesh_boundary_data: dict_utils.KeySequenceDict[model.UACPath]) -> bool:
    """No two segments should overlap in UAC space.

    Builds one polygon per segment from its already-computed, ordered boundary loop
    (`UACConstructor.data.submesh_boundary_data`) and checks pairwise non-overlap -- a geometric
    defect `check_segment_boundary_multiplicity`'s structural check can't see, since two segments
    that correctly *share* a boundary resolve to the exact same computed edge by construction
    (`graph.link` points both at the same node); an overlap here means two segments were given
    UAC targets that collide even though they aren't meant to share territory.

    This deliberately doesn't check for gaps: this anatomy's UAC domain has genuine holes (the PV,
    MV, and LAA openings aren't tiled by any segment), so "the union has no holes" isn't a valid
    generic invariant here -- only "no overlap" is.

    Args:
        submesh_boundary_data: Ordered, UAC-mapped boundary loop per segment
            (`UACData.submesh_boundary_data`).

    Returns:
        `True` if no two segments overlap by more than a small area tolerance. Otherwise prints
        each overlapping pair and returns `False`.
    """
    polygons = {
        key: shapely.Polygon(np.column_stack((boundary.alpha, boundary.beta)))
        for key, boundary in submesh_boundary_data.items()
    }

    is_valid = True
    polygon_keys = list(polygons)
    for i, key_a in enumerate(polygon_keys):
        for key_b in polygon_keys[i + 1 :]:
            overlap_area = polygons[key_a].intersection(polygons[key_b]).area
            if overlap_area > _TILING_AREA_TOLERANCE:
                formatted_key_a = logging.format_key(key_a)
                formatted_key_b = logging.format_key(key_b)
                print(
                    f"Segments {formatted_key_a} and {formatted_key_b} overlap in UAC space "
                    f"(area={overlap_area:.2e})."
                )
                is_valid = False

    return is_valid

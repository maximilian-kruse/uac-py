"""Section 5 of the LA config: `submesh_configs`, regions enclosed by portions of the paths.

See the package docstring (`uacpy.anatomies.left_atrium_4_pvs`) for the chart layout.
"""

from uacpy.anatomies.left_atrium_4_pvs.parameterizations import PV_MITRAL, PV_SEPTAL_LATERAL
from uacpy.construction.model import Segment
from uacpy.construction.names import (
    MITRAL,
    AnatomicalDirection,
    AtrialStructure,
    Axis,
    MitralConnector,
    OstiumLoop,
    OstiumSegment,
    RoofConnector,
    SegmentId,
)

# ==================================================================================================
# 5. SEGMENTS: regions enclosed by portions of the paths
#
# A `Segment` is a region of the mesh bounded by portions of paths: `boundary_paths` lists the
# UAC mappings forming the boundary, and `portions` the (start, end) fraction of each path
# that belongs to this segment (a path shared with a neighbour is used in part, e.g. a PV loop
# is split between its three segments). The boundary loop is closed in chart space; the mesh
# region it encloses is then filled in, and `outside_path` names a path that lies outside the
# region so the fill knows which side of the loop to take. Chart coordinates inside follow
# from the boundary by mean value coordinates, so only boundaries are specified here.
#
# The eight segments of the body (see the module docstring for the layout): roof, anterior,
# septal, posterior_roof / posterior_mv, lateral_roof / lateral_mv, and laa. Each PV adds three
# collar segments, see section 2.
# ==================================================================================================
submesh_configs = {
    # ----------------------------------------------------------------------------------------------
    # Roof: the central square, bounded by the four roof connectors and the four PV loops.
    AtrialStructure.ROOF: Segment(
        boundary_paths=[
            [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
            [AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
            [AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
            [AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
        ],
        portions=[
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ],
        outside_path=[AtrialStructure.MV],
    ),
    # ----------------------------------------------------------------------------------------------
    # Anterior: bottom arm of the cross, between the LSPV and the RSPV and the anterior MV side.
    AnatomicalDirection.ANTERIOR: Segment(
        boundary_paths=[
            [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
            [MITRAL, MitralConnector.LSPV_MV, AnatomicalDirection.ANTERIOR],
            [MITRAL, MitralConnector.RSPV_MV, AnatomicalDirection.ANTERIOR],
            [AtrialStructure.MV, AnatomicalDirection.ANTERIOR],
        ],
        portions=[
            (PV_SEPTAL_LATERAL, PV_MITRAL),
            (PV_SEPTAL_LATERAL, PV_MITRAL),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ],
        outside_path=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # Septal: left arm of the cross, between the RSPV and the RIPV and the septal MV side.
    AnatomicalDirection.SEPTAL: Segment(
        boundary_paths=[
            [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
            [MITRAL, MitralConnector.RIPV_MV, AnatomicalDirection.SEPTAL],
            [MITRAL, MitralConnector.RSPV_MV, AnatomicalDirection.SEPTAL],
            [AtrialStructure.MV, AnatomicalDirection.SEPTAL],
        ],
        portions=[(PV_MITRAL, 1), (PV_MITRAL, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
        outside_path=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # Posterior wall, in two parts split by LAA_POSTERIOR: the part nearer the PV roof ...
    SegmentId.POSTERIOR_ROOF: Segment(
        boundary_paths=[
            [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
            [MITRAL, MitralConnector.RIPV_MV, AnatomicalDirection.POSTERIOR],
            [MITRAL, MitralConnector.LIPV_LAA, AnatomicalDirection.POSTERIOR],
            [MITRAL, MitralConnector.LAA_POSTERIOR],
            [AtrialStructure.LAA, AnatomicalDirection.POSTERIOR],
        ],
        portions=[
            (PV_SEPTAL_LATERAL, PV_MITRAL),
            (PV_SEPTAL_LATERAL, PV_MITRAL),
            (0, 1),
            (0, 1 / 2),
            (0, 1),
            (0, 1),
            (3 / 4, 1),
        ],
        outside_path=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # ... and the part nearer the MV.
    SegmentId.POSTERIOR_MV: Segment(
        boundary_paths=[
            [MITRAL, MitralConnector.RIPV_MV, AnatomicalDirection.POSTERIOR],
            [MITRAL, MitralConnector.LAA_MV, AnatomicalDirection.POSTERIOR],
            [MITRAL, MitralConnector.LAA_POSTERIOR],
            [AtrialStructure.LAA, AnatomicalDirection.POSTERIOR],
            [AtrialStructure.MV, AnatomicalDirection.POSTERIOR],
        ],
        portions=[(1 / 2, 1), (0, 1), (0, 1), (1 / 2, 3 / 4), (0, 1)],
        outside_path=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # Lateral wall, in two parts split by LAA_LATERAL: the part nearer the PV roof ...
    SegmentId.LATERAL_ROOF: Segment(
        boundary_paths=[
            [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            [AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
            [MITRAL, MitralConnector.LIPV_LAA, AnatomicalDirection.LATERAL],
            [MITRAL, MitralConnector.LSPV_MV, AnatomicalDirection.LATERAL],
            [MITRAL, MitralConnector.LAA_LATERAL],
            [AtrialStructure.LAA, AnatomicalDirection.LATERAL],
        ],
        portions=[(PV_MITRAL, 1), (PV_MITRAL, 1), (0, 1), (0, 1), (0, 1 / 2), (0, 1), (0, 1 / 4)],
        outside_path=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # ... and the part nearer the MV.
    SegmentId.LATERAL_MV: Segment(
        boundary_paths=[
            [MITRAL, MitralConnector.LAA_MV, AnatomicalDirection.LATERAL],
            [MITRAL, MitralConnector.LSPV_MV, AnatomicalDirection.LATERAL],
            [MITRAL, MitralConnector.LAA_LATERAL],
            [AtrialStructure.LAA, AnatomicalDirection.LATERAL],
            [AtrialStructure.MV, AnatomicalDirection.LATERAL],
        ],
        portions=[(0, 1), (1 / 2, 1), (0, 1), (1 / 4, 1 / 2), (0, 1)],
        outside_path=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # LAA: the appendage itself, bounded only by its own loop (the square patch).
    SegmentId.LAA: Segment(
        boundary_paths=[[AtrialStructure.LAA, AtrialStructure.LAA]],
        portions=[(0, 1)],
        outside_path=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
    ),
    # ----------------------------------------------------------------------------------------------
    # LIPV (left inferior PV): upper right of the 2x2 PV grid, centre (2, 3/2) in the chart.
    # Three collar segments, one per pair of neighbouring spokes: segment_1 (AP to SL),
    # segment_2 (SL to mitral), segment_3 (mitral to AP). Each uses the matching portion of the
    # antral and ostial loop (`PV_SEPTAL_LATERAL`, `PV_MITRAL`) and the two spokes.
    AtrialStructure.LIPV: {
        OstiumSegment.SEGMENT_1: Segment(
            boundary_paths=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_2: Segment(
            boundary_paths=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LIPV, MITRAL],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_3: Segment(
            boundary_paths=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LIPV, MITRAL],
                [AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
            ],
            portions=[
                (PV_MITRAL, 1),
                (PV_MITRAL, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LSPV (left superior PV): lower right of the 2x2 PV grid, centre (2, 1) in the chart.
    # Three collar segments, one per pair of neighbouring spokes: segment_1 (AP to SL),
    # segment_2 (SL to mitral), segment_3 (mitral to AP). Each uses the matching portion of the
    # antral and ostial loop (`PV_SEPTAL_LATERAL`, `PV_MITRAL`) and the two spokes.
    AtrialStructure.LSPV: {
        OstiumSegment.SEGMENT_1: Segment(
            boundary_paths=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_2: Segment(
            boundary_paths=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LSPV, MITRAL],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_3: Segment(
            boundary_paths=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.LSPV, MITRAL],
                [AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
            ],
            portions=[
                (PV_MITRAL, 1),
                (PV_MITRAL, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RSPV (right superior PV): lower left of the 2x2 PV grid, centre (1, 1) in the chart.
    # Three collar segments, one per pair of neighbouring spokes: segment_1 (AP to SL),
    # segment_2 (SL to mitral), segment_3 (mitral to AP). Each uses the matching portion of the
    # antral and ostial loop (`PV_SEPTAL_LATERAL`, `PV_MITRAL`) and the two spokes.
    AtrialStructure.RSPV: {
        OstiumSegment.SEGMENT_1: Segment(
            boundary_paths=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_2: Segment(
            boundary_paths=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RSPV, MITRAL],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_3: Segment(
            boundary_paths=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RSPV, MITRAL],
                [AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
            ],
            portions=[
                (PV_MITRAL, 1),
                (PV_MITRAL, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RIPV (right inferior PV): upper left of the 2x2 PV grid, centre (1, 3/2) in the chart.
    # Three collar segments, one per pair of neighbouring spokes: segment_1 (AP to SL),
    # segment_2 (SL to mitral), segment_3 (mitral to AP). Each uses the matching portion of the
    # antral and ostial loop (`PV_SEPTAL_LATERAL`, `PV_MITRAL`) and the two spokes.
    AtrialStructure.RIPV: {
        OstiumSegment.SEGMENT_1: Segment(
            boundary_paths=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_2: Segment(
            boundary_paths=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RIPV, MITRAL],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (PV_SEPTAL_LATERAL, PV_MITRAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
        OstiumSegment.SEGMENT_3: Segment(
            boundary_paths=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
                [AtrialStructure.RIPV, MITRAL],
                [AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
            ],
            portions=[
                (PV_MITRAL, 1),
                (PV_MITRAL, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=[AtrialStructure.MV],
        ),
    },
}

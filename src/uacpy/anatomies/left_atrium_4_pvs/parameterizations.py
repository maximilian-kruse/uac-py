"""Section 2 of the LA config: `parameterization_configs`, where markers sit along each path.

See the package docstring (`uacpy.anatomies.left_atrium_4_pvs`) for the chart layout.
"""

from uacpy.construction.model import PathParameterization
from uacpy.construction.names import (
    MITRAL,
    AnatomicalDirection,
    AtrialStructure,
    Axis,
    MitralConnector,
    MitralVariant,
    OstiumLoop,
    RoofConnector,
)

# ==================================================================================================
# 2. PARAMETERIZATIONS: where markers sit along each path
#
# Each path is parameterized by relative arclength in [0, 1]. A `PathParameterization` pins
# markers (single mesh vertices) to chosen relative lengths; the points between them are
# spread evenly by arclength. Section 4 then assigns chart coordinates to those lengths.
#
# On a PV loop the markers are the three spoke markers. Walking the loop from the AP spoke
# (0), the SL spoke is a quarter of the way round and the mitral spoke 5/8 of the way. The
# loop is then closed again at 1. Drawn for the LIPV (the others are mirror images):
#
#                                   ooooooooooooo
#                               oooo             oooo      mitral spoke (5/8)
#                            ooo                     ooo
#                          ooo      2                  //o
#                         oo                         /// oo
#                       oo            :::::::::    ///     oo
#                      oo          :::         :://         oo
#                      o         :::             :::         o
#                     oo        ::                 ::        oo
#                     o         :                   :         o
#     SL spoke (1/4)  -----------                   :         o
#                     o         :                   :         o
#                     oo        ::                 ::        oo
#                      o         :::             :::    3    o
#                      oo          :::         :::          oo
#                       oo     1      ::::|::::            oo
#                         oo              |              oo
#                          ooo            |            ooo
#                            ooo          |          ooo
#                               oooo      |      oooo
#                                   oooooo|oooooo
#                                            AP spoke (0 and 1)
#
#     o antral loop   : ostial loop   1, 2, 3 = segment_1, segment_2, segment_3
#     segment_1 = positions [0, 1/4], segment_2 = [1/4, 5/8], segment_3 = [5/8, 1]
# ==================================================================================================
PV_SEPTAL_LATERAL = 1 / 4  # SL spoke, as a fraction of a PV loop from the AP spoke
PV_MITRAL = 5 / 8  # mitral spoke, same convention


parameterization_configs = {
    # ----------------------------------------------------------------------------------------------
    # LIPV (left inferior PV): upper right of the 2x2 PV grid, centre (2, 3/2) in the chart.
    # Both loops carry the AP, SL and mitral marker at 0, 1/4, 5/8 (so each marker pair on
    # the antral and ostial loop lines up). Each spoke runs from its antral marker (0) to its
    # ostial marker (1).
    AtrialStructure.LIPV: {
        OstiumLoop.ANTRAL: PathParameterization(
            path=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        OstiumLoop.OSTIAL: PathParameterization(
            path=[AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        Axis.ANTERIOR_POSTERIOR: PathParameterization(
            path=[AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        Axis.SEPTAL_LATERAL: PathParameterization(
            path=[AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
        MITRAL: PathParameterization(
            path=[AtrialStructure.LIPV, MITRAL],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LSPV (left superior PV): lower right of the 2x2 PV grid, centre (2, 1) in the chart.
    # Both loops carry the AP, SL and mitral marker at 0, 1/4, 5/8 (so each marker pair on
    # the antral and ostial loop lines up). Each spoke runs from its antral marker (0) to its
    # ostial marker (1).
    AtrialStructure.LSPV: {
        OstiumLoop.ANTRAL: PathParameterization(
            path=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        OstiumLoop.OSTIAL: PathParameterization(
            path=[AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        Axis.ANTERIOR_POSTERIOR: PathParameterization(
            path=[AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        Axis.SEPTAL_LATERAL: PathParameterization(
            path=[AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
        MITRAL: PathParameterization(
            path=[AtrialStructure.LSPV, MITRAL],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RSPV (right superior PV): lower left of the 2x2 PV grid, centre (1, 1) in the chart.
    # Both loops carry the AP, SL and mitral marker at 0, 1/4, 5/8 (so each marker pair on
    # the antral and ostial loop lines up). Each spoke runs from its antral marker (0) to its
    # ostial marker (1).
    AtrialStructure.RSPV: {
        OstiumLoop.ANTRAL: PathParameterization(
            path=[AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        OstiumLoop.OSTIAL: PathParameterization(
            path=[AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        Axis.ANTERIOR_POSTERIOR: PathParameterization(
            path=[AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        Axis.SEPTAL_LATERAL: PathParameterization(
            path=[AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
        MITRAL: PathParameterization(
            path=[AtrialStructure.RSPV, MITRAL],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RIPV (right inferior PV): upper left of the 2x2 PV grid, centre (1, 3/2) in the chart.
    # Both loops carry the AP, SL and mitral marker at 0, 1/4, 5/8 (so each marker pair on
    # the antral and ostial loop lines up). Each spoke runs from its antral marker (0) to its
    # ostial marker (1).
    AtrialStructure.RIPV: {
        OstiumLoop.ANTRAL: PathParameterization(
            path=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        OstiumLoop.OSTIAL: PathParameterization(
            path=[AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL],
        ),
        Axis.ANTERIOR_POSTERIOR: PathParameterization(
            path=[AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        Axis.SEPTAL_LATERAL: PathParameterization(
            path=[AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
        MITRAL: PathParameterization(
            path=[AtrialStructure.RIPV, MITRAL],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL, MITRAL],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LAA loop: four markers at quarters, in order LIPV, lateral, MV, posterior. They are the
    # ends of the LIPV_LAA, LAA_LATERAL, LAA_MV and LAA_POSTERIOR connectors.
    AtrialStructure.LAA: PathParameterization(
        path=[AtrialStructure.LAA],
        markers=[
            [AtrialStructure.LAA, AtrialStructure.LIPV],
            [AtrialStructure.LAA, AnatomicalDirection.LATERAL],
            [AtrialStructure.LAA, AtrialStructure.MV],
            [AtrialStructure.LAA, AnatomicalDirection.POSTERIOR],
        ],
        marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
    ),
    # ----------------------------------------------------------------------------------------------
    # MV loop: four markers at quarters, in order RSPV, LSPV, LAA, RIPV (the ends of the
    # connectors arriving at the MV). They are the four corners of the cross in the chart.
    AtrialStructure.MV: PathParameterization(
        path=[AtrialStructure.MV],
        markers=[
            [AtrialStructure.MV, AtrialStructure.RSPV],
            [AtrialStructure.MV, AtrialStructure.LSPV],
            [AtrialStructure.MV, AtrialStructure.LAA],
            [AtrialStructure.MV, AtrialStructure.RIPV],
        ],
        marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
    ),
    # ----------------------------------------------------------------------------------------------
    # Roof connectors run between the AP markers (LIPV-LSPV, RSPV-RIPV) or the SL markers
    # (LSPV-RSPV, RIPV-LIPV) of the two PVs they join.
    AtrialStructure.ROOF: {
        RoofConnector.LIPV_LSPV: PathParameterization(
            path=[AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        RoofConnector.LSPV_RSPV: PathParameterization(
            path=[AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
        RoofConnector.RSPV_RIPV: PathParameterization(
            path=[AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            ],
            marker_relative_positions=[0, 1],
        ),
        RoofConnector.RIPV_LIPV: PathParameterization(
            path=[AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # A mitral connector that ends on the MV bends around a corner of the cross, so it has two
    # parameterizations ("variants"), one per segment it borders: LSPV_MV anterior and lateral,
    # RIPV_MV septal and posterior. The lateral/posterior variant has an extra marker at 1/2,
    # where LAA_LATERAL / LAA_POSTERIOR end. RSPV_MV, LIPV_LAA and LAA_MV also reach a corner
    # twice, but need one parameterization only: their two copies differ in chart position
    # alone (section 4).
    MITRAL: {
        MitralConnector.LIPV_LAA: PathParameterization(
            path=[MITRAL, MitralConnector.LIPV_LAA],
            markers=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.LAA, AtrialStructure.LIPV],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralConnector.LAA_MV: PathParameterization(
            path=[MITRAL, MitralConnector.LAA_MV],
            markers=[
                [AtrialStructure.LAA, AtrialStructure.MV],
                [AtrialStructure.MV, AtrialStructure.LAA],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralVariant.LSPV_MV_ANTERIOR: PathParameterization(
            path=[MITRAL, MitralConnector.LSPV_MV],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.MV, AtrialStructure.LSPV],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralVariant.LSPV_MV_LATERAL: PathParameterization(
            path=[MITRAL, MitralConnector.LSPV_MV],
            markers=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL, MITRAL],
                [MITRAL, MitralConnector.LSPV_MV],
                [AtrialStructure.MV, AtrialStructure.LSPV],
            ],
            marker_relative_positions=[0, 1 / 2, 1],
        ),
        MitralConnector.RSPV_MV: PathParameterization(
            path=[MITRAL, MitralConnector.RSPV_MV],
            markers=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.MV, AtrialStructure.RSPV],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralVariant.RIPV_MV_SEPTAL: PathParameterization(
            path=[MITRAL, MitralConnector.RIPV_MV],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, MITRAL],
                [AtrialStructure.MV, AtrialStructure.RIPV],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralVariant.RIPV_MV_POSTERIOR: PathParameterization(
            path=[MITRAL, MitralConnector.RIPV_MV],
            markers=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL, MITRAL],
                [MITRAL, MitralConnector.RIPV_MV],
                [AtrialStructure.MV, AtrialStructure.RIPV],
            ],
            marker_relative_positions=[0, 1 / 2, 1],
        ),
        MitralConnector.LAA_LATERAL: PathParameterization(
            path=[MITRAL, MitralConnector.LAA_LATERAL],
            markers=[
                [AtrialStructure.LAA, AnatomicalDirection.LATERAL],
                [MITRAL, MitralConnector.LSPV_MV],
            ],
            marker_relative_positions=[0, 1],
        ),
        MitralConnector.LAA_POSTERIOR: PathParameterization(
            path=[MITRAL, MitralConnector.LAA_POSTERIOR],
            markers=[
                [AtrialStructure.LAA, AnatomicalDirection.POSTERIOR],
                [MITRAL, MitralConnector.RIPV_MV],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
}

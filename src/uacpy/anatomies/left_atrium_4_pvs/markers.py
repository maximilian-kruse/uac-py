"""Section 3 of the LA config: `marker_configs`, single mesh vertices on the paths.

See the package docstring (`uacpy.anatomies.left_atrium_4_pvs`) for the chart layout.
"""

from uacpy.construction.model import Marker, PositionKind
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
# 3. MARKERS: single mesh vertices on those paths
#
# A marker is a vertex addressed through a path. Nearly all are at the start (`position=0`)
# or end (`position=-1`) of a raw path: e.g. the antral SL marker of the LIPV is the end of the
# roof connector arriving there, and each ostial marker is where a spoke ends on the ostial
# loop. The two markers in the MITRAL group are different: they sit halfway (`RELATIVE`, 0.5)
# along the anterior / septal variant of a mitral connector, and are where LAA_LATERAL and
# LAA_POSTERIOR end. Markers are built from paths in section 1, so they have no coordinates.
# ==================================================================================================
marker_configs = {
    # ----------------------------------------------------------------------------------------------
    # LIPV (left inferior PV): upper right of the 2x2 PV grid, centre (2, 3/2) in the chart.
    # Antral markers are ends of roof / mitral connectors, ostial markers are spoke ends.
    AtrialStructure.LIPV: {
        OstiumLoop.ANTRAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[MITRAL, MitralConnector.LIPV_LAA],
                position_type=PositionKind.INDEX,
                position=0,
            ),
        },
        OstiumLoop.OSTIAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[AtrialStructure.LIPV, MITRAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    # LSPV (left superior PV): lower right of the 2x2 PV grid, centre (2, 1) in the chart.
    # Antral markers are ends of roof / mitral connectors, ostial markers are spoke ends.
    AtrialStructure.LSPV: {
        OstiumLoop.ANTRAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
            MITRAL: Marker(
                path=[MITRAL, MitralConnector.LSPV_MV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
        },
        OstiumLoop.OSTIAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[AtrialStructure.LSPV, MITRAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    # RSPV (right superior PV): lower left of the 2x2 PV grid, centre (1, 1) in the chart.
    # Antral markers are ends of roof / mitral connectors, ostial markers are spoke ends.
    AtrialStructure.RSPV: {
        OstiumLoop.ANTRAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[MITRAL, MitralConnector.RSPV_MV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
        },
        OstiumLoop.OSTIAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[AtrialStructure.RSPV, MITRAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    # RIPV (right inferior PV): upper left of the 2x2 PV grid, centre (1, 3/2) in the chart.
    # Antral markers are ends of roof / mitral connectors, ostial markers are spoke ends.
    AtrialStructure.RIPV: {
        OstiumLoop.ANTRAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
            MITRAL: Marker(
                path=[MITRAL, MitralConnector.RIPV_MV],
                position_type=PositionKind.INDEX,
                position=0,
            ),
        },
        OstiumLoop.OSTIAL: {
            Axis.ANTERIOR_POSTERIOR: Marker(
                path=[AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            Axis.SEPTAL_LATERAL: Marker(
                path=[AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
            MITRAL: Marker(
                path=[AtrialStructure.RIPV, MITRAL],
                position_type=PositionKind.INDEX,
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    # LAA loop markers: where the connectors from the LIPV, the MV and the mid-markers reach it.
    AtrialStructure.LAA: {
        AtrialStructure.LIPV: Marker(
            path=[MITRAL, MitralConnector.LIPV_LAA],
            position_type=PositionKind.INDEX,
            position=-1,
        ),
        AtrialStructure.MV: Marker(
            path=[MITRAL, MitralConnector.LAA_MV],
            position_type=PositionKind.INDEX,
            position=0,
        ),
        AnatomicalDirection.LATERAL: Marker(
            path=[MITRAL, MitralConnector.LAA_LATERAL],
            position_type=PositionKind.INDEX,
            position=0,
        ),
        AnatomicalDirection.POSTERIOR: Marker(
            path=[MITRAL, MitralConnector.LAA_POSTERIOR],
            position_type=PositionKind.INDEX,
            position=0,
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # MV loop markers: where the mitral connectors from each PV and the LAA reach the MV.
    AtrialStructure.MV: {
        AtrialStructure.RSPV: Marker(
            path=[MITRAL, MitralConnector.RSPV_MV],
            position_type=PositionKind.INDEX,
            position=-1,
        ),
        AtrialStructure.LSPV: Marker(
            path=[MITRAL, MitralConnector.LSPV_MV],
            position_type=PositionKind.INDEX,
            position=-1,
        ),
        AtrialStructure.LAA: Marker(
            path=[MITRAL, MitralConnector.LAA_MV],
            position_type=PositionKind.INDEX,
            position=-1,
        ),
        AtrialStructure.RIPV: Marker(
            path=[MITRAL, MitralConnector.RIPV_MV],
            position_type=PositionKind.INDEX,
            position=-1,
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # Mid-connector markers, splitting LSPV_MV and RIPV_MV; see section 2.
    MITRAL: {
        MitralConnector.LSPV_MV: Marker(
            path=[MITRAL, MitralVariant.LSPV_MV_ANTERIOR],
            position_type=PositionKind.RELATIVE,
            position=0.5,
        ),
        MitralConnector.RIPV_MV: Marker(
            path=[MITRAL, MitralVariant.RIPV_MV_SEPTAL],
            position_type=PositionKind.RELATIVE,
            position=0.5,
        ),
    },
}

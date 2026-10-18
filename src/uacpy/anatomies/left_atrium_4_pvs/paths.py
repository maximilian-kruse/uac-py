"""Section 1 of the LA config: `path_configs`, the lines that exist on the mesh.

See the package docstring (`uacpy.anatomies.left_atrium_4_pvs`) for the chart layout.
"""

from uacpy.construction.model import BoundaryPath, ConnectionPath, EndpointKind
from uacpy.construction.names import (
    MITRAL,
    AtrialStructure,
    Axis,
    MitralConnector,
    OstiumLoop,
    RoofConnector,
)

# ==================================================================================================
# 1. PATHS: which lines exist on the mesh
#
# Two kinds. A `BoundaryPath` is read off the anatomical tags: the boundary loop of a tagged
# region (PV, LAA, MV). A `ConnectionPath` is the shortest path on the mesh between two such
# loops, or markers on them. To keep paths from merging, a new path may not touch the
# `inadmissible_contact` paths at all, and may touch the `inadmissible_along` paths (e.g.
# start or end there) but not travel along them.
# ==================================================================================================
path_configs = {
    # ----------------------------------------------------------------------------------------------
    # LIPV (left inferior PV): upper right of the 2x2 PV grid, centre (2, 3/2) in the chart.
    # Antral and ostial loop from the tag, plus three spokes: the shortest path from the spoke
    # marker on the antral loop (the end of a roof or mitral connector) to the ostial loop.
    # Each later spoke must stay clear of the earlier ones so that the three never merge.
    AtrialStructure.LIPV: {
        OstiumLoop.ANTRAL: BoundaryPath(
            feature_tag=AtrialStructure.LIPV,
            coincides_with_mesh_boundary=False,
        ),
        OstiumLoop.OSTIAL: BoundaryPath(
            feature_tag=AtrialStructure.LIPV,
            coincides_with_mesh_boundary=True,
        ),
        Axis.ANTERIOR_POSTERIOR: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            end=[AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            ],
        ),
        Axis.SEPTAL_LATERAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            end=[AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[[AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR]],
            inadmissible_along=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            ],
        ),
        MITRAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LIPV, OstiumLoop.ANTRAL, MITRAL],
            end=[AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[
                [AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
            ],
            inadmissible_along=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LSPV (left superior PV): lower right of the 2x2 PV grid, centre (2, 1) in the chart.
    # Antral and ostial loop from the tag, plus three spokes: the shortest path from the spoke
    # marker on the antral loop (the end of a roof or mitral connector) to the ostial loop.
    # Each later spoke must stay clear of the earlier ones so that the three never merge.
    AtrialStructure.LSPV: {
        OstiumLoop.ANTRAL: BoundaryPath(
            feature_tag=AtrialStructure.LSPV,
            coincides_with_mesh_boundary=False,
        ),
        OstiumLoop.OSTIAL: BoundaryPath(
            feature_tag=AtrialStructure.LSPV,
            coincides_with_mesh_boundary=True,
        ),
        Axis.ANTERIOR_POSTERIOR: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            end=[AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            ],
        ),
        Axis.SEPTAL_LATERAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            end=[AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[[AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR]],
            inadmissible_along=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            ],
        ),
        MITRAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.LSPV, OstiumLoop.ANTRAL, MITRAL],
            end=[AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[
                [AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
            ],
            inadmissible_along=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RSPV (right superior PV): lower left of the 2x2 PV grid, centre (1, 1) in the chart.
    # Antral and ostial loop from the tag, plus three spokes: the shortest path from the spoke
    # marker on the antral loop (the end of a roof or mitral connector) to the ostial loop.
    # Each later spoke must stay clear of the earlier ones so that the three never merge.
    AtrialStructure.RSPV: {
        OstiumLoop.ANTRAL: BoundaryPath(
            feature_tag=AtrialStructure.RSPV,
            coincides_with_mesh_boundary=False,
        ),
        OstiumLoop.OSTIAL: BoundaryPath(
            feature_tag=AtrialStructure.RSPV,
            coincides_with_mesh_boundary=True,
        ),
        Axis.ANTERIOR_POSTERIOR: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            end=[AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            ],
        ),
        Axis.SEPTAL_LATERAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RSPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            end=[AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[[AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR]],
            inadmissible_along=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            ],
        ),
        MITRAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RSPV, OstiumLoop.ANTRAL, MITRAL],
            end=[AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[
                [AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
            ],
            inadmissible_along=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RIPV (right inferior PV): upper left of the 2x2 PV grid, centre (1, 3/2) in the chart.
    # Antral and ostial loop from the tag, plus three spokes: the shortest path from the spoke
    # marker on the antral loop (the end of a roof or mitral connector) to the ostial loop.
    # Each later spoke must stay clear of the earlier ones so that the three never merge.
    AtrialStructure.RIPV: {
        OstiumLoop.ANTRAL: BoundaryPath(
            feature_tag=AtrialStructure.RIPV,
            coincides_with_mesh_boundary=False,
        ),
        OstiumLoop.OSTIAL: BoundaryPath(
            feature_tag=AtrialStructure.RIPV,
            coincides_with_mesh_boundary=True,
        ),
        Axis.ANTERIOR_POSTERIOR: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.ANTERIOR_POSTERIOR],
            end=[AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            ],
        ),
        Axis.SEPTAL_LATERAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RIPV, OstiumLoop.ANTRAL, Axis.SEPTAL_LATERAL],
            end=[AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[[AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR]],
            inadmissible_along=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            ],
        ),
        MITRAL: ConnectionPath(
            boundary_types=[EndpointKind.MARKER, EndpointKind.PATH],
            start=[AtrialStructure.RIPV, OstiumLoop.ANTRAL, MITRAL],
            end=[AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            inadmissible_contact=[
                [AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
                [AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
            ],
            inadmissible_along=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LAA ostium: boundary loop of the LAA tag. It lies in the mesh interior, since the
    # appendage continues past it.
    AtrialStructure.LAA: BoundaryPath(
        feature_tag=AtrialStructure.LAA,
        coincides_with_mesh_boundary=False,
    ),
    # ----------------------------------------------------------------------------------------------
    # MV annulus: boundary loop of the MV tag. It coincides with the mesh boundary (the open edge
    # where the mesh was cut at the valve).
    AtrialStructure.MV: BoundaryPath(
        feature_tag=AtrialStructure.MV,
        coincides_with_mesh_boundary=True,
    ),
    # ----------------------------------------------------------------------------------------------
    # Roof connectors: shortest paths between the antral loops of neighbouring PVs, going around
    # LIPV -> LSPV -> RSPV -> RIPV -> LIPV. Together with the loops they enclose the roof segment.
    AtrialStructure.ROOF: {
        RoofConnector.LIPV_LSPV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            ],
        ),
        RoofConnector.LSPV_RSPV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.LSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            ],
        ),
        RoofConnector.RSPV_RIPV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.RSPV, OstiumLoop.ANTRAL],
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            ],
        ),
        RoofConnector.RIPV_LIPV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            inadmissible_contact=None,
            inadmissible_along=[
                [AtrialStructure.RIPV, OstiumLoop.ANTRAL],
                [AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # Mitral connectors: LIPV -> LAA, LAA -> MV, and one from each of LSPV, RSPV, RIPV to the MV.
    # LAA_LATERAL and LAA_POSTERIOR run from the LAA ostium to a marker halfway along the
    # LSPV_MV and RIPV_MV connector, so they split the LA wall between the PVs and the MV.
    MITRAL: {
        MitralConnector.LIPV_LAA: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.LAA],
            inadmissible_contact=None,
            inadmissible_along=[[AtrialStructure.LIPV, OstiumLoop.ANTRAL], [AtrialStructure.LAA]],
        ),
        MitralConnector.LAA_MV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.LAA],
            end=[AtrialStructure.MV],
            inadmissible_contact=None,
            inadmissible_along=[[AtrialStructure.LAA], [AtrialStructure.MV]],
        ),
        MitralConnector.LSPV_MV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.MV],
            inadmissible_contact=(
                [AtrialStructure.LAA],
                [MITRAL, MitralConnector.LIPV_LAA],
                [MITRAL, MitralConnector.LAA_MV],
            ),
            inadmissible_along=[[AtrialStructure.LSPV, OstiumLoop.ANTRAL], [AtrialStructure.MV]],
        ),
        MitralConnector.RSPV_MV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.MV],
            inadmissible_contact=None,
            inadmissible_along=[[AtrialStructure.RSPV, OstiumLoop.ANTRAL], [AtrialStructure.MV]],
        ),
        MitralConnector.RIPV_MV: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.PATH],
            start=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            end=[AtrialStructure.MV],
            inadmissible_contact=([MITRAL, MitralConnector.RSPV_MV],),
            inadmissible_along=[[AtrialStructure.RIPV, OstiumLoop.ANTRAL], [AtrialStructure.MV]],
        ),
        MitralConnector.LAA_LATERAL: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.MARKER],
            start=[AtrialStructure.LAA],
            end=[MITRAL, MitralConnector.LSPV_MV],
            inadmissible_contact=([AtrialStructure.MV],),
            inadmissible_along=[[AtrialStructure.LAA], [MITRAL, MitralConnector.LSPV_MV]],
        ),
        MitralConnector.LAA_POSTERIOR: ConnectionPath(
            boundary_types=[EndpointKind.PATH, EndpointKind.MARKER],
            start=[AtrialStructure.LAA],
            end=[MITRAL, MitralConnector.RIPV_MV],
            inadmissible_contact=([AtrialStructure.MV],),
            inadmissible_along=[[AtrialStructure.LAA], [MITRAL, MitralConnector.RIPV_MV]],
        ),
    },
}

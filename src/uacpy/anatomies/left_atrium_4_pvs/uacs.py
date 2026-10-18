"""Section 4 of the LA config: `uac_configs`, chart coordinates along each path.

See the package docstring (`uacpy.anatomies.left_atrium_4_pvs`) for the chart layout.
"""

import numpy as np

from uacpy.anatomies.left_atrium_4_pvs.parameterizations import PV_MITRAL, PV_SEPTAL_LATERAL
from uacpy.construction.model import UACMapping
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
# 4. UACS: chart coordinates along each path
#
# Chart coordinates: assigning (alpha, beta) to the points on each path. `relative_positions`
# are the arclength fractions of the path; the `uacs` are the chart points they map to, with
# straight lines between. Chart units are arbitrary: only the layout matters.
#
#     2.5                  .#########################.*****************
#                          .                         **               *
#                          .                       ** *               *
#                          .                     **   *               *
#                          .....................**    *      LAA      *
#     2.0                  .                      **  *               *
#                          .                        ***               *
#                          .                         .*               *
#                          ooooooo             ooooooo*****************
#         ................oo     oo           oo     oo...**       **..
#     1.5 #               o  RI   -------------  LI   o     **   **   #
#         #               oo     oo           oo     oo       ***     #
#         #                ooo-ooo    roof     ooo-ooo         .      #
#         #                ooo-ooo             ooo-ooo         .      #
#         #               oo     oo           oo     oo        .      #
#     1.0 #               o  RS   -------------  LS   o        .      #
#         ................oo     oo           oo     oo................
#                          ooooooo             ooooooo
#                          .                         .
#                          .                         .
#     0.5                  .                         .
#                          .                         .
#                          .                         .
#                          .                         .
#                          .                         .
#     0.0                  .#########################.
#         +-------------------+-------------------+-------------------+
#         0                   1                   2                   3
#
#     #  MV annulus      .  mitral connectors      -  roof connectors
#     o  PV antral loop  *  LAA ostium (V-notches) and LAA patch (square)
#
# The numbers below place the PVs in the 2x2 grid, set the loop radii, and size the LAA patch,
# its notches and the part of a mitral connector after it turns the corner.
# ==================================================================================================
LIPV_CENTER = (2, 3 / 2)  # PV centres: the 2x2 grid in the middle of the chart
LSPV_CENTER = (2, 1)
RIPV_CENTER = (1, 3 / 2)
RSPV_CENTER = (1, 1)
LAA_CENTER = (13 / 5, 21 / 10)  # centre of the square LAA patch
PV_INNER_RADIUS = 1 / 5  # antral loop radius
PV_OUTER_RADIUS = 1 / 10  # ostial loop radius (the hole at the PV opening)
LAA_LENGTH = 4 / 5  # side of the LAA patch
LAA_DISTANCE = 1 / 4  # spacing of the points on the notch edge, along the MV side
LAA_DEPTH = 1 / 4  # how far the V-shaped notch reaches into the body
MITRAL_LENGTH = 1  # distance from a PV centre to the MV side it faces


# ==================================================================================================
uac_configs = {
    # ----------------------------------------------------------------------------------------------
    # LIPV (left inferior PV): upper right of the 2x2 PV grid, centre (2, 3/2) in the chart.
    # Antral loop -> circle of radius 1/5, ostial loop -> circle of radius 1/10; the spokes
    # are radial lines between them.
    # Spokes point S (AP), W (SL) and NE (mitral); the loops run clockwise from AP.
    AtrialStructure.LIPV: {
        OstiumLoop.ANTRAL: UACMapping(
            path=[AtrialStructure.LIPV, OstiumLoop.ANTRAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        OstiumLoop.OSTIAL: UACMapping(
            path=[AtrialStructure.LIPV, OstiumLoop.OSTIAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
                (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
                (
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        Axis.ANTERIOR_POSTERIOR: UACMapping(
            path=[AtrialStructure.LIPV, Axis.ANTERIOR_POSTERIOR],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        Axis.SEPTAL_LATERAL: UACMapping(
            path=[AtrialStructure.LIPV, Axis.SEPTAL_LATERAL],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
        MITRAL: UACMapping(
            path=[AtrialStructure.LIPV, MITRAL],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LSPV (left superior PV): lower right of the 2x2 PV grid, centre (2, 1) in the chart.
    # Antral loop -> circle of radius 1/5, ostial loop -> circle of radius 1/10; the spokes
    # are radial lines between them.
    # Spokes point N (AP), W (SL) and SE (mitral); the loops run counter-clockwise from AP.
    AtrialStructure.LSPV: {
        OstiumLoop.ANTRAL: UACMapping(
            path=[AtrialStructure.LSPV, OstiumLoop.ANTRAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        OstiumLoop.OSTIAL: UACMapping(
            path=[AtrialStructure.LSPV, OstiumLoop.OSTIAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
                (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
                (
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        Axis.ANTERIOR_POSTERIOR: UACMapping(
            path=[AtrialStructure.LSPV, Axis.ANTERIOR_POSTERIOR],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        Axis.SEPTAL_LATERAL: UACMapping(
            path=[AtrialStructure.LSPV, Axis.SEPTAL_LATERAL],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
            ],
        ),
        MITRAL: UACMapping(
            path=[AtrialStructure.LSPV, MITRAL],
            relative_positions=[0, 1],
            uacs=[
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RSPV (right superior PV): lower left of the 2x2 PV grid, centre (1, 1) in the chart.
    # Antral loop -> circle of radius 1/5, ostial loop -> circle of radius 1/10; the spokes
    # are radial lines between them.
    # Spokes point N (AP), E (SL) and SW (mitral); the loops run clockwise from AP.
    AtrialStructure.RSPV: {
        OstiumLoop.ANTRAL: UACMapping(
            path=[AtrialStructure.RSPV, OstiumLoop.ANTRAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                (
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        OstiumLoop.OSTIAL: UACMapping(
            path=[AtrialStructure.RSPV, OstiumLoop.OSTIAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
                (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
                (
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        Axis.ANTERIOR_POSTERIOR: UACMapping(
            path=[AtrialStructure.RSPV, Axis.ANTERIOR_POSTERIOR],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        Axis.SEPTAL_LATERAL: UACMapping(
            path=[AtrialStructure.RSPV, Axis.SEPTAL_LATERAL],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        MITRAL: UACMapping(
            path=[AtrialStructure.RSPV, MITRAL],
            relative_positions=[0, 1],
            uacs=[
                (
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # RIPV (right inferior PV): upper left of the 2x2 PV grid, centre (1, 3/2) in the chart.
    # Antral loop -> circle of radius 1/5, ostial loop -> circle of radius 1/10; the spokes
    # are radial lines between them.
    # Spokes point S (AP), E (SL) and NW (mitral); the loops run counter-clockwise from AP.
    AtrialStructure.RIPV: {
        OstiumLoop.ANTRAL: UACMapping(
            path=[AtrialStructure.RIPV, OstiumLoop.ANTRAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        OstiumLoop.OSTIAL: UACMapping(
            path=[AtrialStructure.RIPV, OstiumLoop.OSTIAL],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_MITRAL, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
                (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
                (
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        Axis.ANTERIOR_POSTERIOR: UACMapping(
            path=[AtrialStructure.RIPV, Axis.ANTERIOR_POSTERIOR],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        Axis.SEPTAL_LATERAL: UACMapping(
            path=[AtrialStructure.RIPV, Axis.SEPTAL_LATERAL],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
            ],
        ),
        MITRAL: UACMapping(
            path=[AtrialStructure.RIPV, MITRAL],
            relative_positions=[0, 1],
            uacs=[
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # LAA, three mappings of the same loop. `LAA`: the whole loop is the square patch (used by
    # the `laa` segment). `lateral` and `posterior`: the ostium arcs of the loop, drawn as the
    # V-shaped notches that the neighbouring segments see (see the drawing).
    AtrialStructure.LAA: {
        AtrialStructure.LAA: UACMapping(
            path=[AtrialStructure.LAA],
            relative_positions=[0, 1 / 4, 1 / 2, 3 / 4, 1],
            uacs=[
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
                (LAA_CENTER[0] + LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
                (LAA_CENTER[0] + LAA_LENGTH / 2, LAA_CENTER[1] + LAA_LENGTH / 2),
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] + LAA_LENGTH / 2),
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
            ],
        ),
        AnatomicalDirection.LATERAL: UACMapping(
            path=[AtrialStructure.LAA],
            relative_positions=[0, 1 / 4, 1 / 2],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
        AnatomicalDirection.POSTERIOR: UACMapping(
            path=[AtrialStructure.LAA],
            relative_positions=[1 / 2, 3 / 4, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                ),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # MV annulus -> the four straight sides of the cross, one mapping per side over consecutive
    # quarters of the loop: anterior (bottom), lateral (right), posterior (top), septal (left).
    # Consecutive sides do not meet: the empty corner between two sides is a seam, see the module
    # docstring.
    AtrialStructure.MV: {
        AnatomicalDirection.ANTERIOR: UACMapping(
            path=[AtrialStructure.MV],
            relative_positions=[0, 1 / 4],
            uacs=[
                (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
            ],
        ),
        AnatomicalDirection.LATERAL: UACMapping(
            path=[AtrialStructure.MV],
            relative_positions=[1 / 4, 1 / 2],
            uacs=[
                (LSPV_CENTER[0] + MITRAL_LENGTH, LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                (LIPV_CENTER[0] + MITRAL_LENGTH, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
        AnatomicalDirection.POSTERIOR: UACMapping(
            path=[AtrialStructure.MV],
            relative_positions=[1 / 2, 3 / 4],
            uacs=[
                (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + MITRAL_LENGTH),
                (RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + MITRAL_LENGTH),
            ],
        ),
        AnatomicalDirection.SEPTAL: UACMapping(
            path=[AtrialStructure.MV],
            relative_positions=[3 / 4, 1],
            uacs=[
                (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # Roof connectors -> the straight lines between the antral circles of neighbouring PVs.
    AtrialStructure.ROOF: {
        RoofConnector.LIPV_LSPV: UACMapping(
            path=[AtrialStructure.ROOF, RoofConnector.LIPV_LSPV],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        RoofConnector.LSPV_RSPV: UACMapping(
            path=[AtrialStructure.ROOF, RoofConnector.LSPV_RSPV],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        RoofConnector.RSPV_RIPV: UACMapping(
            path=[AtrialStructure.ROOF, RoofConnector.RSPV_RIPV],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        RoofConnector.RIPV_LIPV: UACMapping(
            path=[AtrialStructure.ROOF, RoofConnector.RIPV_LIPV],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    # Mitral connectors. A spoke marker on a PV sits at 45 degrees on the antral circle, hence the
    # offsets of PV_INNER_RADIUS / sqrt(2). A connector that ends on the MV is mapped once per
    # segment it borders (the entries keyed by direction); both copies start at the same point
    # on the PV but end at different MV corners: this is the seam.
    MITRAL: {
        MitralConnector.LIPV_LAA: {
            AnatomicalDirection.LATERAL: UACMapping(
                path=[MITRAL, MitralConnector.LIPV_LAA],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            AnatomicalDirection.POSTERIOR: UACMapping(
                path=[MITRAL, MitralConnector.LIPV_LAA],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                    ),
                ],
            ),
        },
        MitralConnector.LAA_MV: {
            AnatomicalDirection.LATERAL: UACMapping(
                path=[MITRAL, MitralConnector.LAA_MV],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + MITRAL_LENGTH,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            AnatomicalDirection.POSTERIOR: UACMapping(
                path=[MITRAL, MitralConnector.LAA_MV],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                    ),
                    (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + MITRAL_LENGTH),
                ],
            ),
        },
        MitralConnector.LSPV_MV: {
            AnatomicalDirection.ANTERIOR: UACMapping(
                path=[MITRAL, MitralVariant.LSPV_MV_ANTERIOR],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            AnatomicalDirection.LATERAL: UACMapping(
                path=[MITRAL, MitralVariant.LSPV_MV_LATERAL],
                relative_positions=[0, 1 / 2, 1],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + MITRAL_LENGTH / 2,
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LSPV_CENTER[0] + MITRAL_LENGTH,
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
        },
        MitralConnector.RSPV_MV: {
            AnatomicalDirection.ANTERIOR: UACMapping(
                path=[MITRAL, MitralConnector.RSPV_MV],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            AnatomicalDirection.SEPTAL: UACMapping(
                path=[MITRAL, MitralConnector.RSPV_MV],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
        },
        MitralConnector.RIPV_MV: {
            AnatomicalDirection.SEPTAL: UACMapping(
                path=[MITRAL, MitralVariant.RIPV_MV_SEPTAL],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
            AnatomicalDirection.POSTERIOR: UACMapping(
                path=[MITRAL, MitralVariant.RIPV_MV_POSTERIOR],
                relative_positions=[0, 1 / 2, 1],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + MITRAL_LENGTH / 2,
                    ),
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + MITRAL_LENGTH,
                    ),
                ],
            ),
        },
        MitralConnector.LAA_LATERAL: UACMapping(
            path=[MITRAL, MitralConnector.LAA_LATERAL],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + MITRAL_LENGTH / 2,
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
        MitralConnector.LAA_POSTERIOR: UACMapping(
            path=[MITRAL, MitralConnector.LAA_POSTERIOR],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                ),
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + MITRAL_LENGTH / 2,
                ),
            ],
        ),
    },
}

"""Construction-pipeline config for one left-atrium topology: 4 PVs + LAA + MV.

Written out as plain nested dicts of `model` dataclasses, one module per config kind; the entries
refer to each other by key sequence. Every identifier used as a dict key or in a key sequence is a
`StrEnum` member from `uacpy.construction.names`, so a typo is an `AttributeError` at import time,
not a silently-wrong new dict branch.

The idea
--------
The LA surface is cut open along the mitral valve (MV) annulus and flattened into an
(alpha, beta) chart with alpha in [0, 3] and beta in [0, 2.5] (the ranges are specific to this
anatomy). The chart is a cross-shaped domain plus a separate square patch for the left atrial
appendage (LAA). The four pulmonary veins (PVs) sit in a 2x2 grid in the middle. Lines (paths)
between PVs, LAA and MV cut the rest of the atrium into segments:

    2.5                  .#########################.*****************
                         .                         **               *
                         .      posterior_mv     ** *               *
                         .                     **   *               *
                         .....................**    *      laa      *
    2.0                  .                      **  *               *
                         .     posterior_roof     ***               *
                         .                         .*               *
                         ooooooo             ooooooo*****************
        ................oo     oo           oo     oo...**       **..
    1.5 #               o RIPV  ------------- LIPV  o     **   **   #
        #               oo     oo           oo     oo       ***     #
        #   septal       ooo-ooo    roof     ooo-ooo  l_roof .      #
        #                ooo-ooo             ooo-ooo         .      #
        #               oo     oo           oo     oo        .l_mv  #
    1.0 #               o RSPV  ------------- LSPV  o        .      #
        ................oo     oo           oo     oo................
                         ooooooo             ooooooo
                         .                         .
                         .                         .
    0.5                  .        anterior         .
                         .                         .
                         .                         .
                         .                         .
                         .                         .
    0.0                  .#########################.
        +-------------------+-------------------+-------------------+
        0                   1                   2                   3

    #  MV annulus, mapped onto the four straight sides of the cross
    .  mitral connectors: shortest paths from a PV or the LAA to the MV (and the two LAA ones)
    -  roof connectors: shortest paths between neighbouring PVs
    o  PV antral loop, radius 1/5 (the PV opening, the ostial loop, is the hole at its centre)
    *  LAA: the square patch is the appendage itself, the V-shaped notches are its ostium
    l_ lateral_ (segment names are abbreviated in the drawing)

Seams: the chart is a cut-open atrium, so some mesh paths appear twice. The LSPV -> MV connector
is one path on the mesh, but is drawn once going down to the anterior side of the MV and once
going right to its lateral side (`MitralVariant`); the empty corner between the two copies is the
cut. The same holds at the three other corners, which is why the cross has empty corners.

How the five dicts fit together (an entry refers to earlier ones by key sequence):

    paths.py              path_configs              which lines exist on the mesh (tag boundaries,
                                                    shortest paths)
    parameterizations.py  parameterization_configs  where markers sit along each line (relative
                                                    arclength)
    markers.py            marker_configs            markers: single mesh vertices on those lines
    uacs.py               uac_configs               chart coordinates assigned along each line
    segments.py           submesh_configs           segments: regions enclosed by portions of
                                                    those lines

Names: AP = anterior-posterior, SL = septal-lateral. A PV has an antral loop (in the mesh
interior, around the opening) and an ostial loop (on the mesh boundary, where the vein was cut off).
Three spokes (AP, SL and "mitral", pointing towards the MV) join the two loops and cut the
PV's collar into three segments.
"""

from uacpy.anatomies.left_atrium_4_pvs.markers import marker_configs
from uacpy.anatomies.left_atrium_4_pvs.parameterizations import parameterization_configs
from uacpy.anatomies.left_atrium_4_pvs.paths import path_configs
from uacpy.anatomies.left_atrium_4_pvs.segments import submesh_configs
from uacpy.anatomies.left_atrium_4_pvs.uacs import uac_configs

__all__ = [
    "marker_configs",
    "parameterization_configs",
    "path_configs",
    "submesh_configs",
    "uac_configs",
]

"""Interactive pyvista plots for UAC meshes, vector fields, and the segmentation.

Every function here opens a blocking plot window and returns nothing. UAC submeshes are drawn
flat in the (alpha, beta) plane; the original mesh is drawn in patient space.
"""

import numpy as np
import pyvista as pv

from uacpy import mapper
from uacpy.common import dict_utils
from uacpy.construction import model


# ==================================================================================================
def plot_uac_mesh(uac_submesh_data: dict_utils.KeySequenceDict[model.UACSubmesh]) -> None:
    """Plot all UAC submeshes together as wireframes in (alpha, beta) space.

    Args:
        uac_submesh_data: `UACSubmesh` per segment (`UACData.uac_submesh_data`). Entries that
            are `None` are skipped.
    """
    plotter = pv.Plotter()
    for uac_submesh in uac_submesh_data.values():
        if uac_submesh is None:
            continue
        submesh = uac_submesh.to_polydata()
        plotter.add_mesh(submesh, style="wireframe", color="grey")
    plotter.view_xy()
    plotter.show()


# --------------------------------------------------------------------------------------------------
def plot_uac_mesh_with_vector_field(
    uac_submesh_data: dict_utils.KeySequenceDict[model.UACSubmesh],
    vector_field_data: mapper.FieldDict,
    scaling_factor: float,
    vector_color: str = "blue",
    mesh_color: str = "lightgray",
) -> None:
    """Plot UAC submeshes with a per-cell 2D vector field drawn as arrows at the cell centers.

    Args:
        uac_submesh_data: `UACSubmesh` per segment. Entries that are `None` are skipped.
        vector_field_data: Dict with the same keys as `uac_submesh_data`, each value an array of
            shape `(n_cells, 2)` in UAC space. Entries that are `None` are skipped.
        scaling_factor: Arrow length per unit vector magnitude, in (alpha, beta) units.
        vector_color: Color of the arrows.
        mesh_color: Color of the wireframe.
    """
    plotter = pv.Plotter()
    for key, uac_submesh in uac_submesh_data.items():
        vector_field = vector_field_data[key]
        if uac_submesh is None or vector_field is None:
            continue
        submesh = uac_submesh.to_polydata()
        cell_centers = submesh.cell_centers()
        cell_centers.point_data["vector_field_to_plot"] = np.hstack(
            (vector_field, np.zeros((vector_field.shape[0], 1)))
        )
        glyphs = cell_centers.glyph(
            orient="vector_field_to_plot",
            scale=False,
            factor=scaling_factor,
            geom=pv.Arrow(),
        )
        plotter.add_mesh(glyphs, color=vector_color)
        plotter.add_mesh(submesh, style="wireframe", color=mesh_color)
    plotter.view_xy()
    plotter.show()


# --------------------------------------------------------------------------------------------------
def plot_mesh_with_vector_field(
    mesh: pv.PolyData,
    vector_field: np.ndarray,
    scaling_factor: float,
    vector_color: str = "blue",
    mesh_color: str = "lightgray",
) -> None:
    """Plot a surface mesh with a per-cell 3D vector field drawn as arrows at the cell centers.

    Args:
        mesh: Surface mesh in patient space.
        vector_field: Per-cell vectors, shape `(n_cells, 3)`.
        scaling_factor: Arrow length per unit vector magnitude, in mesh units.
        vector_color: Color of the arrows.
        mesh_color: Color of the surface.
    """
    cell_centers = mesh.cell_centers()
    cell_centers.point_data["vector_field_to_plot"] = vector_field
    glyphs = cell_centers.glyph(
        orient="vector_field_to_plot",
        scale=False,
        factor=scaling_factor,
        geom=pv.Arrow(),
    )
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, color=vector_color)
    plotter.add_mesh(mesh, color=mesh_color)
    plotter.show()


# --------------------------------------------------------------------------------------------------
def plot_mesh_with_segmentation(
    marker_data: dict_utils.KeySequenceDict[int],
    uac_path_data: dict_utils.KeySequenceDict[model.UACPath],
    mesh: pv.PolyData,
) -> None:
    """Plot the segmentation (markers and arclength-parameterized paths) on the original mesh.

    Args:
        marker_data: Marker vertex index per marker (`UACData.marker_data`), drawn as red
            spheres. Entries that are `None` are skipped.
        uac_path_data: `UACPath` per path (`UACData.uac_path_data`), drawn as points colored
            by relative arclength. Entries that are `None` are skipped.
        mesh: Original surface mesh in patient space.
    """
    plotter = pv.Plotter(window_size=[700, 500])
    plotter.add_mesh(mesh, style="surface", show_edges=True, color="grey")

    for marker in marker_data.values():
        if marker is None:
            continue
        marker_mesh = pv.PolyData(mesh.points[marker])
        plotter.add_mesh(marker_mesh, color="red", point_size=15, render_points_as_spheres=True)
    for path in uac_path_data.values():
        if path is None:
            continue
        path_mesh = pv.PolyData(mesh.points[path.inds])
        path_mesh.point_data["relative_lengths"] = path.relative_lengths
        plotter.add_mesh(
            path_mesh, point_size=10, scalars="relative_lengths", render_points_as_spheres=True
        )
    plotter.show()

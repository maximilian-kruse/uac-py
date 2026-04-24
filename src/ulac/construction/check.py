import numpy as np
import pyvista as pv

from ulac.common import dict_utils
from ulac.construction import constructor


# ==================================================================================================
def check_uac_mesh_validity(uac_submesh: constructor.UACSubmeshDict) -> bool:
    key_sequence = dict_utils.nested_dict_keys(uac_submesh)

    for key in key_sequence:
        submesh_data = dict_utils.get_dict_entry(key, uac_submesh)
        vertex_coordinates = np.hstack(
            [
                submesh_data.alpha[:, None],
                submesh_data.beta[:, None],
                np.zeros((submesh_data.alpha.shape[0], 1)),
            ]
        )
        simplicies = submesh_data.connectivity
        pv_submesh = pv.PolyData.from_regular_faces(vertex_coordinates, simplicies)
        quality_data = pv_submesh.cell_quality(quality_measure="area")
        cell_areas = quality_data.cell_data["area"]

        if np.any(cell_areas <= 0):
            print(
                f"Foldover detected in submesh {key}: found {np.sum(cell_areas <= 0)} cells with "
                "non-positive area."
            )
            return False

    return True

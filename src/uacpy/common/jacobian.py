"""Per-triangle affine Jacobian between a 3D embedding and a 2D parameterization, via Cramer's rule.

Each triangle's 3D-vertex-to-2D-vertex map is treated as exactly affine (`v_3d = J @ v_2d + b`),
so `J` is fully determined by the triangle's two edge vectors in each space; solving the resulting
2x2 linear system per row via Cramer's rule avoids forming/inverting a matrix per triangle.
"""

import numpy as np


# ==================================================================================================
def compute_cellwise_jacobians(
    vertices_3d: np.ndarray, vertices_2d: np.ndarray, simplices: np.ndarray
) -> np.ndarray:
    """Compute the affine Jacobian of each triangle's 2D-to-3D embedding.

    Args:
        vertices_3d: Vertex positions in 3D, shape `(n_vertices, 3)`.
        vertices_2d: Vertex positions in the 2D parameterization, shape `(n_vertices, 2)`.
        simplices: Triangle vertex indices shared by both point sets, shape `(n_triangles, 3)`.
            Triangles must be non-degenerate in 2D, or the Cramer's-rule denominator vanishes.

    Returns:
        Array of shape `(n_triangles, 3, 2)` whose `[i]`-th entry maps a 2D tangent-plane vector
        on triangle `i` to its 3D counterpart.
    """
    edge_3d_one = vertices_3d[simplices[:, 1]] - vertices_3d[simplices[:, 0]]
    edge_3d_two = vertices_3d[simplices[:, 2]] - vertices_3d[simplices[:, 0]]
    edge_2d_one = vertices_2d[simplices[:, 1]] - vertices_2d[simplices[:, 0]]
    edge_2d_two = vertices_2d[simplices[:, 2]] - vertices_2d[simplices[:, 0]]
    jacobians = np.empty((len(simplices), 3, 2))

    for row_ind in range(3):
        jacobians[:, row_ind, 0], jacobians[:, row_ind, 1] = _compute_jacobian_rows(
            edge_3d_one, edge_3d_two, edge_2d_one, edge_2d_two, row_ind
        )
    return jacobians


# --------------------------------------------------------------------------------------------------
def _compute_jacobian_rows(
    edge_3d_one: np.ndarray,
    edge_3d_two: np.ndarray,
    edge_2d_one: np.ndarray,
    edge_2d_two: np.ndarray,
    row_ind: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one row of every triangle's Jacobian via Cramer's rule.

    Solves `edge_3d[:, row_ind] = J_row @ edge_2d` for both edges of every triangle at once.
    """
    j_col_one = (
        edge_3d_one[:, row_ind] * edge_2d_two[:, 1] - edge_3d_two[:, row_ind] * edge_2d_one[:, 1]
    ) / (edge_2d_one[:, 0] * edge_2d_two[:, 1] - edge_2d_one[:, 1] * edge_2d_two[:, 0])
    j_col_two = (
        edge_3d_one[:, row_ind] * edge_2d_two[:, 0] - edge_3d_two[:, row_ind] * edge_2d_one[:, 0]
    ) / (edge_2d_two[:, 0] * edge_2d_one[:, 1] - edge_2d_two[:, 1] * edge_2d_one[:, 0])
    return j_col_one, j_col_two

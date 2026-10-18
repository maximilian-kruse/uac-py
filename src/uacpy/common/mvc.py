"""Floater's mean value coordinates: a discrete-harmonic map fixing given boundary values.

Floater, M. S. (2003), "Mean value coordinates", Computer Aided Geometric Design 20(1), 19-27.
Each interior vertex's coordinate is the weighted average of its one-ring neighbors', with the
weights (`_compute_weight_matrix`) fixed by the mesh's own geometry; solving `(W - I) x = 0` on
the interior subject to the given `boundary_coordinates` gives a map that reproduces those exactly
on the boundary and interpolates smoothly (no fold-overs, for a convex or otherwise well-shaped
boundary) inside.
"""

import numpy as np
import scipy.sparse as sps


# ==================================================================================================
def compute_mean_value_coordinates(
    vertices: np.ndarray,
    simplices: np.ndarray,
    boundary_inds: np.ndarray,
    boundary_coordinates: np.ndarray,
) -> np.ndarray:
    """Extend 2D boundary coordinates over a mesh's interior by mean value interpolation.

    Args:
        vertices: Vertex positions in 3D, shape `(n_vertices, 3)`.
        simplices: Triangle vertex indices, shape `(n_triangles, 3)`.
        boundary_inds: Indices into `vertices` of the boundary vertices with prescribed values,
            shape `(n_boundary,)`.
        boundary_coordinates: Prescribed 2D coordinates at `boundary_inds`, shape
            `(n_boundary, 2)`.

    Returns:
        Array of shape `(n_vertices, 2)`, exactly equal to `boundary_coordinates` at
        `boundary_inds` and mean-value-interpolated elsewhere. Rows of vertices not referenced by
        `simplices` are left at zero.
    """
    weight_matrix = _compute_weight_matrix(vertices, simplices)
    system_matrix = weight_matrix - sps.eye(weight_matrix.shape[0], format="csr")
    interior_inds = np.setdiff1d(np.unique(simplices), boundary_inds)
    system_matrix_interior = system_matrix[np.ix_(interior_inds, interior_inds)]
    system_matrix_boundary = system_matrix[np.ix_(interior_inds, boundary_inds)]
    rhs_vector_x = system_matrix_boundary @ boundary_coordinates[:, 0]
    rhs_vector_y = system_matrix_boundary @ boundary_coordinates[:, 1]
    uacs_interior_x = sps.linalg.spsolve(system_matrix_interior, -rhs_vector_x)
    uacs_interior_y = sps.linalg.spsolve(system_matrix_interior, -rhs_vector_y)
    uacs_interior = np.hstack([uacs_interior_x[:, None], uacs_interior_y[:, None]])
    uacs = np.zeros((vertices.shape[0], 2))
    uacs[interior_inds] = uacs_interior
    uacs[boundary_inds] = boundary_coordinates

    return uacs


def _compute_weight_matrix(vertices: np.ndarray, simplices: np.ndarray) -> sps.csr_matrix:
    """Build the row-normalized mean value weight matrix `W`.

    Each triangle contributes a weight to both of its non-source vertices, from each of its three
    vertices in turn (`_compute_weight_contribs`); summing those contributions per (source,
    neighbor) pair over every triangle in the one-ring, then dividing each row by its own sum,
    gives Floater's weights directly (no separate normalization step needed elsewhere).
    """
    num_vertices_per_simplex = 3
    list_of_weights = []

    for i in range(num_vertices_per_simplex):
        source_vertex = simplices[:, i]
        neighbor_vertex_one = simplices[:, (i + 1) % num_vertices_per_simplex]
        neighbor_vertex_two = simplices[:, (i + 2) % num_vertices_per_simplex]
        source_coordinates = vertices[source_vertex]
        neighbor_one_coordinates = vertices[neighbor_vertex_one]
        neighbor_two_coordinates = vertices[neighbor_vertex_two]
        weight_contrib_one, weight_contrib_two = _compute_weight_contribs(
            source_coordinates, neighbor_one_coordinates, neighbor_two_coordinates
        )
        weights_array_one = np.hstack(
            [weight_contrib_one[:, None], source_vertex[:, None], neighbor_vertex_one[:, None]]
        )
        weights_array_two = np.hstack(
            [weight_contrib_two[:, None], source_vertex[:, None], neighbor_vertex_two[:, None]]
        )
        list_of_weights.append(weights_array_one)
        list_of_weights.append(weights_array_two)

    weight_contrib_array = np.vstack(list_of_weights)
    weight_matrix = _assemble_weight_matrix(weight_contrib_array)
    return weight_matrix


def _compute_weight_contribs(
    source_coordinates: np.ndarray,
    neighbor_one_coordinates: np.ndarray,
    neighbor_two_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute each triangle's contribution to the mean value weights of the edges at `source`.

    Floater's weight is `tan(theta/2) / |edge|` for the half-angle `theta` this triangle subtends
    at `source` between the edge and its neighbor; `1e-6`-clamping `angle_cosine` away from +-1
    avoids a division blow-up for near-degenerate (near-flat or near-zero-angle) triangles.
    """
    edge_one = neighbor_one_coordinates - source_coordinates
    edge_two = neighbor_two_coordinates - source_coordinates
    edge_one_length = np.linalg.norm(edge_one, axis=1)
    edge_two_length = np.linalg.norm(edge_two, axis=1)
    angle_cosine = np.clip(
        np.einsum("ij,ij->i", edge_one, edge_two) / (edge_one_length * edge_two_length),
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )
    half_angle_tangens = np.sqrt((1.0 - angle_cosine) / (1.0 + angle_cosine))
    weight_contrib_one = half_angle_tangens / edge_one_length
    weight_contrib_two = half_angle_tangens / edge_two_length
    return weight_contrib_one, weight_contrib_two


def _assemble_weight_matrix(weight_contribs: np.ndarray) -> sps.coo_matrix:
    """Scatter per-triangle `(weight, source, neighbor)` contributions into a row-normalized matrix.

    Multiple triangles contribute to the same (source, neighbor) pair (each shared edge is seen
    from both of its adjacent triangles), so `sum_duplicates` combines them before the per-row
    normalization that turns raw weights into Floater's coordinates.
    """
    weight_matrix = sps.coo_matrix(
        (
            weight_contribs[:, 0],
            (weight_contribs[:, 1].astype(int), weight_contribs[:, 2].astype(int)),
        ),
    )
    weight_matrix.sum_duplicates()
    weight_matrix = weight_matrix.tocsr()
    column_sums = np.array(weight_matrix.sum(axis=1)).flatten()
    column_sum_matrix = sps.diags(1 / column_sums)
    weight_matrix = column_sum_matrix @ weight_matrix
    return weight_matrix

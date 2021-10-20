# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import OrderedDict, defaultdict
import numpy as np
import torch
from sty import fg, rs


def load_ansys_mesh(filename):
    import pyansys
    # archive = pyansys.Archive(f"assets/{filename}.cdb", force_linear=True,)
    archive = pyansys.read_binary(filename)
    print(archive)
    mesh = archive.mesh
    print(mesh)
    elems = mesh.elem
    tets = []
    nnum = mesh.nnum.tolist()
    for elem in elems:
        # Get nodes belonging to elem, which are in fields 10-30
        # NOTE: Vector of length 8 is returned. Only 4 nodes are unique.
        elem_nodes = elem[10:18]

        # Get 4 unique nodes in original order
        _, indices = np.unique(elem_nodes, return_index=True)
        indices = sorted(indices)
        elem_nodes = elem_nodes[indices]

        assert (len(elem_nodes) == 4)
        if 0 in elem_nodes:
            continue
        tets.append([nnum.index(elem_nodes[i]) for i in range(4)])
    tets = np.asarray(tets)
    vertices = mesh.nodes
    indices = np.reshape(tets, (-1, ))
    return indices, vertices


def load_mesh(filename):
    import meshio
    mesh = meshio.read(filename)
    vertices = mesh.points
    indices = np.reshape(mesh.cells[0].data, (-1, ))
    return indices, vertices


def as_tensor(var, device='cuda', requires_grad=True):
    if type(var) == torch.Tensor:
        return var
    return torch.tensor(var, device=device, dtype=torch.float32, requires_grad=requires_grad)


def convert_lame(young, poisson):
    # converts Young's modulus and Poisson ratio to Lamé parameters μ and λ
    return (young / (2 * (1.0 + poisson)), (young * poisson) / ((1.0 + poisson) * (1.0 - 2 * poisson)))


def convert_viz_indices(faces):
    # bring N indices into pyvista format [N, i, j, k]
    return np.hstack(np.hstack([[[faces.shape[1]]] * faces.shape[0], faces]))


class MeshTopology:
    """
    Helper functions for manipulating tetrahedral meshes.
    """
    def __init__(self, tet_indices):
        self.elements_per_face = defaultdict(set)
        self.elements_per_edge = defaultdict(set)
        self.elements_per_node = defaultdict(set)
        # use ordered dict as ordered set
        self.unique_faces = OrderedDict()
        self.unique_edges = OrderedDict()
        self.unique_nodes = OrderedDict()
        for e, tet in enumerate(tet_indices):
            self.elements_per_node[tet[0]].add(e)
            self.elements_per_node[tet[1]].add(e)
            self.elements_per_node[tet[2]].add(e)
            self.elements_per_node[tet[3]].add(e)
            for edge in MeshTopology.edge_indices(tet):
                self.elements_per_edge[edge].add(e)
                self.unique_edges[edge] = None
            for face in MeshTopology.face_indices(tet):
                self.elements_per_face[face].add(e)
                self.unique_faces[face] = None

    @staticmethod
    def face_indices(tet):
        face1 = tuple(sorted((tet[0], tet[2], tet[1])))
        face2 = tuple(sorted((tet[1], tet[2], tet[3])))
        face3 = tuple(sorted((tet[0], tet[1], tet[3])))
        face4 = tuple(sorted((tet[0], tet[3], tet[2])))
        return (face1, face2, face3, face4)

    @staticmethod
    def edge_indices(tet):
        edge1 = tuple(sorted((tet[0], tet[1])))
        edge2 = tuple(sorted((tet[1], tet[2])))
        edge3 = tuple(sorted((tet[2], tet[0])))
        edge4 = tuple(sorted((tet[0], tet[3])))
        edge5 = tuple(sorted((tet[1], tet[3])))
        edge6 = tuple(sorted((tet[2], tet[3])))
        return (edge1, edge2, edge3, edge4, edge5, edge6)

    def surface_faces(self):
        return [face for face in self.unique_faces if len(self.elements_per_face[face]) == 1]

    def surface_edges(self):
        edges = set()
        faces = self.surface_faces()
        for face in faces:
            edges.add(tuple(sorted((face[0], face[1]))))
            edges.add(tuple(sorted((face[1], face[2]))))
            edges.add(tuple(sorted((face[2], face[0]))))
        return edges

    def surface_nodes(self):
        nodes = set()
        faces = self.surface_faces()
        for face in faces:
            nodes.add(face[0])
            nodes.add(face[1])
            nodes.add(face[2])
        return nodes


def get_mesh_aabb(vertices):
    lo = np.min(vertices, axis=0)
    hi = np.max(vertices, axis=0)
    corners = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ])
    indices = np.array([0, 2, 1, 3, 2, 0, 1, 2, 6, 6, 5, 1, 3, 7, 6,
                       6, 2, 3, 0, 1, 5, 5, 4, 0, 0, 4, 7, 7, 3, 0, 4, 5, 6, 6, 7, 4])
    return corners, indices


def colored(text, color):
    print(color + text + rs.fg)


# apply boundary conditions
def inside(box, p):
    if p[0] < box.min[0] or p[0] > box.max[0]:
        return False
    if p[1] < box.min[1] or p[1] > box.max[1]:
        return False
    if p[2] < box.min[2] or p[2] > box.max[2]:
        return False
    return True

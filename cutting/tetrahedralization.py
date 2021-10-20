# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import sys

import wildmeshing as wm

from .utils import load_mesh, MeshTopology, convert_viz_indices

import meshio
from PyQt5 import Qt

import pyvista as pv
from pyvistaqt import BackgroundPlotter


def tetrahedralize(filename, y_is_up=False, scaling=0.0005, stop_quality=10, max_its=50, edge_length_r=0.1, epsilon=0.01, visualize=True):

    indices, vertices = load_mesh(filename)
    if not y_is_up:
        vz = vertices[:, 1].copy()
        vertices[:, 1] = vertices[:, 2].copy()
        vertices[:, 2] = vz
    vertices *= scaling

    tetra = wm.Tetrahedralizer(
        stop_quality=stop_quality, max_its=max_its, edge_length_r=edge_length_r, epsilon=epsilon)
    tetra.set_mesh(vertices, np.array(indices).reshape(-1, 3))

    tetra.tetrahedralize()
    VT, TT = tetra.get_tet_mesh()

    print(
        f"Tetrahedralized cross section to {TT.shape[0]} elements and {VT.shape[0]} vertices.")

    if visualize:
        top = MeshTopology(TT)
        unique_edges_mesh = pv.PolyData(VT, convert_viz_indices(
            np.array(list(top.unique_edges.keys()))))

        app = Qt.QApplication(sys.argv)
        plotter = BackgroundPlotter(title='Tetrahedralizer', auto_update=20.)
        plotter.show()

        plotter.set_viewup([0., 1., 0.])
        plotter.set_focus([0., 0., 0.])
        plotter.set_position([0.0, .2, -0.01], True)
        plotter.add_axes_at_origin(labels_off=True)

        plotter.add_mesh(unique_edges_mesh, style="wireframe",
                         render_lines_as_tubes=True, line_width=3.5, color='cyan')

        # add ground plane
        plotter.add_mesh(pv.Plane(direction=(0, 1, 0), i_size=100,
                         j_size=100, i_resolution=10, j_resolution=10), color='white')

    mesh = meshio.Mesh(VT, [("quad", TT)])
    target_filename = filename.replace('.obj', '.msh')
    meshio.write(target_filename, mesh)
    print(f"Saved tetrahedralized mesh to {target_filename}.")

    if visualize:
        app.exec_()

    return True


if __name__ == "__main__":
    filename = "assets/usc_apple3.obj"
    y_is_up = False
    scaling = 0.0005
    tetrahedralize(filename, y_is_up, scaling)

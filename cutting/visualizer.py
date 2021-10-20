# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# fmt: off
import sys
import torch
import copy

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5 import Qt
from threading import Thread

import numpy as np

import pyvista as pv
from pyvistaqt import BackgroundPlotter

from matplotlib.colors import ListedColormap
# fmt: on


class Visualizer:
    """
    Basic interactive visualizer using pyvista.
    """

    def __init__(self,
                 sim,
                 render_frequency=50,
                 show_static_vertices=True,
                 show_dependent_particles=True,
                 show_cut_spring_sides=False,
                 show_cut_virtual_triangles=True,
                 plot_knife_force_history=True,
                 skip_steps=1,
                 scaling=500.,
                 show_ground_plane=True,
                 show_knife_mesh_normals=False,
                 show_mesh_wireframe=False,
                 knife_force_history_limit=20.):

        self.render_frequency = render_frequency

        self.plotter = BackgroundPlotter(title='DiSECt', auto_update=20.)
        self.plotter.show()
        self.skip_steps = skip_steps
        self.sim_step = 0
        self.last_rx = None
        self.screen_label = self.plotter.add_text("", font_size=10)
        self.anim_menu = self.plotter.main_menu.addMenu('Simulation')
        self.scaling = scaling
        self.unit = 1e-2 * self.scaling

        action = Qt.QAction('Play / Pause', self.plotter.app_window)
        action.setShortcut(' ')
        action.triggered.connect(self.toggle_play)
        self.anim_menu.addAction(action)

        action = Qt.QAction('Stop', self.plotter.app_window)
        action.setShortcut('Backspace')
        action.triggered.connect(self.stop)
        self.anim_menu.addAction(action)

        self.anim_menu.addSeparator()
        self.is_playing = True
        self.show_static_vertices = show_static_vertices
        self.show_dependent_particles = show_dependent_particles

        # mesh = pv.Arrow()
        # self.plotter.add_mesh(mesh)
        # mesh = examples.load_hexbeam()
        # self.plotter.add_mesh(mesh, show_edges=True, color='white')

        self.sim = sim
        if sim is None:
            # knife mesh
            spine_dim = 2.
            edge_dim = 0.8
            spine_height = 4.
            tip_height = 0.4
            depth = 8.

            vertices = np.array([
                [-spine_dim / 2, spine_height / 2, -depth / 2],
                [spine_dim / 2, spine_height / 2, -depth / 2],
                [-edge_dim / 2, -spine_height / 2, -depth / 2],
                [edge_dim / 2, -spine_height / 2, -depth / 2],
                [0, -spine_height / 2 - tip_height, -depth / 2],
                [-spine_dim / 2, spine_height / 2, depth / 2],
                [spine_dim / 2, spine_height / 2, depth / 2],
                [-edge_dim / 2, -spine_height / 2, depth / 2],
                [edge_dim / 2, -spine_height / 2, depth / 2],
                [0, -spine_height / 2 - tip_height, depth / 2],
            ])

            # mesh faces
            faces = np.hstack([
                [4, 0, 1, 3, 2],
                [3, 2, 3, 4],
                [4, 6, 5, 7, 8],
                [3, 8, 7, 9],
                [4, 1, 6, 8, 3],
                [4, 3, 8, 9, 4],
                [4, 0, 5, 6, 1],
                [4, 5, 0, 2, 7],
                [4, 7, 2, 4, 9],
            ])
        else:
            # disable autodiff
            sim.disable_gradients()
            vertices = np.array(self.sim.builder.particle_q)
            faces = np.array(self.sim.builder.tri_indices)
            if len(faces) == 0:
                faces = np.zeros((0, 4), dtype=np.int)
            else:
                faces = Visualizer.convert_tri_indices(faces)

        if len(faces) > 0:
            self.surf = pv.PolyData(vertices, faces)
            if hasattr(self.sim, "viz_scalars") and self.sim.viz_scalars is not None:
                self.surf.point_arrays["scalars"] = self.sim.viz_scalars
                self.surf.set_active_scalars('scalars')
                self.plotter.add_mesh(
                    self.surf, show_edges=True, color='white', scalars='scalars', opacity=0.85)
            else:
                self.plotter.add_mesh(
                    self.surf, show_edges=True, color='white', opacity=0.85)
        else:
            self.surf = None
        # self.plotter.add_mesh(mesh, show_edges=True, color='white')
        self.plotter.add_axes_at_origin(labels_off=True)

        self.cut_spring_sides = None
        self.cut_virtual_tris_above_cut = None
        self.cut_virtual_tris_below_cut = None
        self.cut_triangles = None

        self.rigids = []
        self.geo_src = []    # original model.shape_geo_src.vertices
        self.geo_type = []
        self.geo_scale = []
        self.hist_knife_force = []
        self.hist_time = []
        self.knife_normals = None
        if sim is not None:
            if show_static_vertices:
                masses = np.array(self.sim.builder.particle_mass)
                static_vertices = vertices[masses == 0.0] * self.scaling
                if len(static_vertices) > 0:
                    static_vertices_mesh = pv.PolyData(
                        np.array(static_vertices))
                    self.plotter.add_mesh(static_vertices_mesh, style='points',
                                          render_points_as_spheres=True, point_size=self.unit, color='black')

            if show_dependent_particles and len(self.sim.builder.dependent_particle_indices) > 0:
                dependent_particle_indices = np.array(
                    self.sim.builder.dependent_particle_indices)
                dependent_xs = np.array(self.sim.builder.particle_q)[
                    dependent_particle_indices[:, 1]] * self.scaling
                self.dependent_vertices = pv.PolyData(np.array(dependent_xs))
                self.plotter.add_mesh(self.dependent_vertices, style='points',
                                      render_points_as_spheres=True, point_size=2.5 * self.unit, color='purple')

            if show_knife_mesh_normals:
                knife_tris = np.array(
                    self.sim.builder.knife_tri_indices).reshape((-1, 3))
                knife_verts = np.array(
                    self.sim.builder.knife_tri_vertices) * self.scaling
                knife_points = knife_verts[knife_tris]
                knife_centroids = np.mean(knife_points, axis=1)
                sides_ab = knife_points[:, 1] - knife_points[:, 0]
                sides_ac = knife_points[:, 2] - knife_points[:, 0]
                knife_normals = np.cross(sides_ab, sides_ac)
                knife_normals /= np.linalg.norm(knife_normals,
                                                axis=1)[:, np.newaxis]
                knife_normal_points = np.concatenate(
                    [knife_centroids, knife_centroids + knife_normals], axis=0)
                knife_normal_mesh_indices = np.zeros(
                    (len(knife_tris), 3), dtype=np.int)
                knife_normal_mesh_indices[:, 0] = 2
                knife_normal_mesh_indices[:, 1] = np.arange(len(knife_tris))
                knife_normal_mesh_indices[:, 2] = np.arange(
                    len(knife_tris)) + len(knife_tris)
                self.knife_normals_src = knife_normal_points.copy()
                # knife_normal_points *= self.scaling
                self.knife_normals = pv.PolyData(
                    knife_normal_points, knife_normal_mesh_indices)
                self.plotter.add_mesh(self.knife_normals,
                                      style='wireframe', color='cyan')

            if self.sim.model is None:
                print("Visualizer triggered model creation because it was undefined")
                self.sim.create_model()

            self.cut_spring_indices = self.sim.model.cut_spring_indices.detach().cpu().numpy()
            self.cut_edge_indices = self.sim.model.cut_edge_indices.detach().cpu().numpy()
            if self.cut_spring_indices.shape[0] > 0:
                cut_vertices = np.zeros(
                    (self.sim.model.cut_edge_indices.shape[0], 3))
                cut_spring_indices = np.hstack(
                    np.hstack([[[2]] * self.cut_spring_indices.shape[0], self.cut_spring_indices]))
                self.cut_springs = pv.PolyData(
                    cut_vertices, cut_spring_indices)
                cut_spring_stiffness = self.sim.model.cut_spring_stiffness.detach().cpu().numpy()
                self.max_cut_spring_stiffness = np.max(cut_spring_stiffness)
                colormap = ListedColormap(
                    [[1., 0.3, 0., 0.], [1., 0.3, 0., 1.]])
                self.plotter.add_mesh(self.cut_springs,
                                      scalars=cut_spring_stiffness,
                                      style='wireframe',
                                      line_width=5.,
                                      render_lines_as_tubes=True,
                                      clim=[0., self.max_cut_spring_stiffness],
                                      cmap=colormap)

                cut_spring_side_vertices = np.zeros(
                    (self.cut_spring_indices.shape[0] * 4, 3))
                cut_spring_side_scalars = np.array(
                    [0, 0, 1, 1] * self.cut_spring_indices.shape[0])
                cut_spring_side_indices = np.hstack(
                    [[2, i * 2, i * 2 + 1] for i in range(self.cut_spring_indices.shape[0] * 2)])

                if show_cut_spring_sides:
                    self.cut_spring_sides = pv.PolyData(
                        cut_spring_side_vertices, cut_spring_side_indices)
                    self.plotter.add_mesh(self.cut_spring_sides,
                                          scalars=cut_spring_side_scalars,
                                          cmap=["red", "blue"],
                                          style='wireframe',
                                          line_width=2,
                                          render_lines_as_tubes=True,
                                          clim=[0., 1])

                if self.sim.model.cut_virtual_tri_indices.shape[0] > 0:
                    self.cut_virtual_tri_indices = self.sim.model.cut_virtual_tri_indices.detach().cpu().numpy()
                    # cut_virtual_tri_indices = Visualizer.convert_tri_indices(self.cut_virtual_tri_indices)
                    if show_cut_virtual_triangles:
                        cut_virtual_tri_indices_above_cut = Visualizer.convert_tri_indices(
                            self.sim.model.cut_virtual_tri_indices_above_cut.detach().cpu().numpy())
                        self.cut_virtual_tris_above_cut = pv.PolyData(
                            cut_vertices, cut_virtual_tri_indices_above_cut)
                        self.plotter.add_mesh(self.cut_virtual_tris_above_cut,
                                              show_edges=True,
                                              edge_color='magenta',
                                              color='magenta',
                                              line_width=2,
                                              render_lines_as_tubes=False)
                        cut_virtual_tri_indices_below_cut = Visualizer.convert_tri_indices(
                            self.sim.model.cut_virtual_tri_indices_below_cut.detach().cpu().numpy())
                        self.cut_virtual_tris_below_cut = pv.PolyData(
                            cut_vertices, cut_virtual_tri_indices_below_cut)
                        self.plotter.add_mesh(self.cut_virtual_tris_below_cut,
                                              show_edges=True,
                                              edge_color='yellow',
                                              color='yellow',
                                              line_width=2,
                                              render_lines_as_tubes=False)

                if self.sim.model.cut_tri_indices.shape[0] > 0:
                    cut_tri_vertices = np.vstack((vertices, cut_vertices))
                    cut_tri_indices = self.sim.model.cut_tri_indices.detach().cpu().numpy()
                    cut_tri_indices = Visualizer.convert_tri_indices(
                        cut_tri_indices)
                    self.cut_triangles = pv.PolyData(
                        cut_tri_vertices, cut_tri_indices)
                    self.plotter.add_mesh(
                        self.cut_triangles, show_edges=True, color='white', opacity=0.85)

            if show_ground_plane and sim.model.ground:
                self.plotter.add_mesh(pv.Plane(direction=(
                    0, 1, 0), i_size=100, j_size=100, i_resolution=10, j_resolution=10), color='white')

            geo_types = self.sim.model.shape_geo_type.detach().cpu()
            self.geo_colors = ("brown", "green", "orange",
                               "peru", "olive", "wheat", "sienna")
            for i in range(len(geo_types)):
                geo_type = geo_types[i]
                link_id = self.sim.builder.shape_body[i]
                self.geo_type.append(geo_type)
                scale = self.sim.model.shape_geo_scale[i].detach(
                ).cpu().numpy()
                self.geo_scale.append(scale)
                if geo_type == 0:
                    shape = pv.Sphere(radius=scale[0] * self.scaling)
                    m = self.plotter.add_mesh(
                        shape, color=self.geo_colors[i % len(self.geo_colors)], smooth_shading=True)
                    self.rigids.append(shape)
                    self.geo_src.append(np.copy(shape.points))
                elif geo_type == 1:
                    shape = pv.Box(bounds=[
                        -scale[0] / 2 * self.scaling,
                        scale[0] / 2 * self.scaling,
                        -scale[1] / 2 * self.scaling,
                        scale[1] / 2 * self.scaling,
                        -scale[2] / 2 * self.scaling,
                        scale[2] / 2 * self.scaling
                    ])
                    m = self.plotter.add_mesh(
                        shape, color=self.geo_colors[i % len(self.geo_colors)])
                    self.rigids.append(shape)
                    self.geo_src.append(np.copy(shape.points))
                elif geo_type == 2:
                    # render capsule as cylinder
                    shape = pv.Cylinder(
                        radius=scale[0] * self.scaling, height=scale[1] * self.scaling * 2, direction=(0, 0, 1))
                    m = self.plotter.add_mesh(
                        shape, color=self.geo_colors[i % len(self.geo_colors)], smooth_shading=True)
                    self.rigids.append(shape)
                    self.geo_src.append(np.copy(shape.points))
                elif geo_type == 3:
                    src = self.sim.model.shape_geo_src[i]
                    shape = pv.PolyData(np.array(src.vertices) * self.scaling,
                                        Visualizer.convert_tri_indices(np.array(src.indices).reshape((-1, 3))))
                    m = self.plotter.add_mesh(shape,
                                              color=self.geo_colors[i % len(
                                                  self.geo_colors)] if link_id != self.sim.builder.knife_link_index else 'orange',
                                              show_edges=show_mesh_wireframe,
                                              smooth_shading=False,
                                              line_width=1,
                                              edge_color='black',
                                              )
                    #   opacity=0.85)
                    self.rigids.append(shape)
                    self.geo_src.append(np.copy(shape.points))
                elif geo_type == 6:
                    pass
                else:
                    print("Visualization for shape type %i has not been implemented." %
                          geo_type, file=sys.stderr)

            self.sim.state = self.sim.model.state()
            self.start_state = self.sim.model.state()
            self.start_model = copy.copy(self.sim.model)
            self.start()

        self.plotter.set_viewup([0., 1., 0.])
        self.plotter.set_focus([0., 0., 0.])
        self.plotter.set_position([0., .05, 0.2], True)
        print("Created visualizer")

        if sim is not None and plot_knife_force_history:
            import time
            max_steps = int(knife_force_history_limit / self.sim.sim_dt)
            # Plot knife force history

            def plot_force():
                import matplotlib.pyplot as plt
                plt.ion()
                fig, ax = plt.subplots()
                fig.canvas.set_window_title('Knife Force [N]')
                while True:
                    hist_knife_force = np.copy(self.hist_knife_force)
                    hist_time = np.copy(self.hist_time)
                    # make sure both arrays have the same length
                    if hist_time.shape[0] > hist_knife_force.shape[0]:
                        hist_time = hist_time[:hist_knife_force.shape[0]]
                    elif hist_time.shape[0] < hist_knife_force.shape[0]:
                        hist_knife_force = hist_knife_force[:hist_time.shape[0]]
                    if len(hist_knife_force) > max_steps:
                        hist_knife_force = hist_knife_force[-max_steps:]
                        hist_time = self.hist_time[-max_steps:]
                    ax.clear()
                    ax.grid()
                    ax.plot(hist_time, hist_knife_force, color="C0")
                    plt.pause(0.0001)
                    time.sleep(0.2)

            self.plot_thread = Thread(target=plot_force)
            self.plot_thread.start()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        print("Play?", self.is_playing)
        if self.sim is None:
            return
        if self.is_playing:
            self.start()

    def update_view(self):
        import dflex as df
        ps = self.sim.state.particle_q.detach().cpu().numpy() * self.scaling
        if self.surf is not None:
            self.surf.points = ps
            if hasattr(self.sim, "viz_scalars") and self.sim.viz_scalars is not None:
                self.surf.point_arrays["scalars"] = self.sim.viz_scalars
        rx = self.sim.state.body_X_sm[:,
                                      :3].detach().cpu().numpy() * self.scaling
        for i in range(len(self.rigids)):
            body_id = self.sim.builder.shape_body[i]
            rot = df.quat_to_matrix(
                self.sim.state.body_X_sm[body_id, 3:7].detach().cpu().numpy())
            pos = rx[body_id]
            # self.rigids[i].points = self.geo_scale[i][:3] * (rot @ self.geo_src[i].T).T + pos
            self.rigids[i].points = (rot @ self.geo_src[i].T).T + pos
            if self.last_rx is not None:
                # draw traces
                self.plotter.add_lines(np.array(
                    (self.last_rx[body_id], pos)), color=self.geo_colors[i % len(self.geo_colors)] if body_id != self.sim.model.knife_link_index else 'orange')
            if self.knife_normals is not None and body_id == self.sim.model.knife_link_index:
                # self.knife_normals.points = self.geo_scale[i][:3] * (rot @ self.knife_normals_src.T).T + pos
                self.knife_normals.points = (
                    rot @ self.knife_normals_src.T).T + pos
        self.last_rx = rx.copy()

        if self.show_dependent_particles and len(self.sim.builder.dependent_particle_indices) > 0:
            dependent_particle_indices = np.array(
                self.sim.builder.dependent_particle_indices)
            dependent_xs = ps[dependent_particle_indices[:, 1]]
            self.dependent_vertices.points = dependent_xs

        if hasattr(self.sim.state, 'knife_f') and self.sim.state.knife_f is not None and len(self.hist_knife_force) > 0:
            self.screen_label.SetText(
                1, f"Total knife force: {self.hist_knife_force[-1]:.3f}\nMax knife force: {torch.max(torch.norm(self.sim.state.knife_f, dim=1)).item():.3f}")

        # update cutting elements
        if self.cut_spring_indices.shape[0] > 0:
            cut_edge_coords = self.sim.model.cut_edge_coords.detach().cpu().numpy()
            cut_spring_stiffness = self.sim.state.cut_spring_ke.detach().cpu().numpy()
            cut_vertices = ps[self.cut_edge_indices[:, 0]] * (1.0 - cut_edge_coords)[
                :, None] + ps[self.cut_edge_indices[:, 1]] * cut_edge_coords[:, None]

            if self.cut_triangles is not None:
                self.cut_triangles.points = np.vstack((ps, cut_vertices))
            self.cut_springs.points = cut_vertices
            self.plotter.update_scalars(
                cut_spring_stiffness, mesh=self.cut_springs, render=False)
            self.screen_label.SetText(
                3, "Mean Spring Stiffness: %.4f" % np.mean(cut_spring_stiffness))

        if self.cut_virtual_tris_above_cut is not None:
            self.cut_virtual_tris_above_cut.points = cut_vertices
        if self.cut_virtual_tris_below_cut is not None:
            self.cut_virtual_tris_below_cut.points = cut_vertices

        knife_pos = self.sim.state.body_X_sm[self.sim.model.knife_link_index].detach().cpu().numpy()[
            :7].tolist()
        knife_vel = self.sim.state.body_v_s[self.sim.model.knife_link_index].detach().cpu().numpy()[
            :6].tolist()
        self.screen_label.SetText(2, "Time: %02.2f\nKnife pos: %.3f %.3f %.3f   %.3f %.3f %.3f %.3f\nKnife vel: %.3f %.3f %.3f   %.3f %.3f %.3f" % (
            self.sim.sim_time, knife_pos[0], knife_pos[1], knife_pos[2], knife_pos[3], knife_pos[4], knife_pos[5], knife_pos[6], knife_vel[0], knife_vel[1], knife_vel[2], knife_vel[3], knife_vel[4], knife_vel[5]))

        self.screen_label.SetText(
            0,
            f"Mesh: {self.sim.settings.generators[self.sim.settings.generator].filename}\nVertices: {len(self.sim.builder.particle_q)}\nElements: {len(self.sim.builder.tet_indices)}\nCutting springs: {len(self.sim.builder.cut_spring_indices)}\nIntegrator: {self.sim.settings.integrator}  dt={self.sim.settings.sim_dt}\nMode: {self.sim.settings.cutting.mode}\nKnife type: {self.sim.knife.type}"
        )

    def start(self):
        def simulate():
            import dflex as df
            df.config.no_grad = True
            with torch.no_grad():
                self.coarse_sim_step = 0
                while self.is_playing:
                    # forward dynamics
                    for _ in range(self.skip_steps + 1):
                        self.sim.simulation_step()
                        self.sim_step += 1

                    if self.coarse_sim_step % self.render_frequency == 0:
                        if hasattr(self.sim.state, 'knife_f'):
                            knife_f = torch.sum(torch.norm(
                                self.sim.state.knife_f.detach(), dim=1)).item()
                            self.hist_knife_force.append(knife_f)
                        self.hist_time.append(self.sim.sim_time)
                        self.update_view()
                    self.coarse_sim_step += 1

        self.thread = Thread(target=simulate)
        self.thread.start()

        # def step():
        #     for _ in range(self.skip_steps + 1):
        #         self.sim.state = self.sim.integrator.forward(self.sim.model, self.sim.state, self.sim.sim_dt)
        #         self.sim.sim_time += self.sim.sim_dt
        #         self.sim_step += 1
        #     self.update_view()
        # self.plotter.add_callback(step, interval=1)

    def stop(self):
        self.is_playing = False
        self.sim.model = copy.copy(self.start_model)
        self.sim.state = self.sim.model.state()
        # self.sim.state = copy.copy(self.start_state)
        self.sim.sim_time = 0.0
        self.sim_step = 0
        self.update_view()

    @staticmethod
    def convert_tri_indices(faces):
        # bring triangle indices into pyvista format [3, i, j, k]
        return np.hstack(np.hstack([[[3]] * faces.shape[0], faces]))


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = Visualizer(None)
    sys.exit(app.exec_())

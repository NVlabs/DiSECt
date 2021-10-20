# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""The main class of the differentiable cutting simulator DiSECt."""

# fmt: off
import torch
import copy
import pickle
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scipy.spatial import KDTree

from cutting.settings import default_parameters, datasets
from cutting.utils import *
from cutting.plot_posterior import colorline
from cutting.knife import *
from cutting.motion import *
from cutting.urdf_loader import load_urdf
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sty import fg, rs
# fmt: on


class CuttingSim:
    def __init__(self,
                 settings: dict,
                 dataset: str = '',
                 log_folder: str = 'log',
                 experiment_name: str = 'cutting_experiment',
                 parameters={},
                 force_all_params_shared=False,
                 adapter: str = 'cuda',
                 requires_grad=True,
                 show_cutting_surface=False):
        assert isinstance(settings, dict)
        self.settings = settings
        self.adapter = adapter

        self.dataset = dataset
        if dataset != '':
            dataset_entry = datasets[dataset]
            self.parameters = {key: copy.deepcopy(
                default_parameters[key]) for key in dataset_entry.params if key in default_parameters}
            print(
                f"Using dataset \"{dataset}\" with parameters {', '.join(self.parameters.keys())}.")
        else:
            self.parameters = copy.deepcopy(parameters)
        self.settings["dataset"] = dataset
        # print(json.dumps(settings, indent=4))

        if force_all_params_shared:
            print("Setting all parameters as non-individual (shared).")
            for p in self.parameters.values():
                p.individual = False
        try:
            Path(log_folder).mkdir(parents=True, exist_ok=True)
            print(f'Using log folder at "{os.path.abspath(log_folder)}".')
        except:
            pass
        self.log_folder = os.path.abspath(log_folder)

        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)

        self.sim_duration = settings.sim_duration          # 5.0       # seconds
        self.sim_substeps = settings.sim_substeps          # 1000      # 32
        self.sim_substeps = max(1, self.sim_substeps)
        self.sim_dt = settings.sim_dt
        self.sim_steps = int(self.sim_duration / self.sim_dt)
        self.sim_coarse_dt = self.sim_dt * self.sim_substeps
        self.sim_coarse_steps = self.sim_steps // self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        self.experiment_name = experiment_name

        self.motion = ConstantLinearVelocityMotion(initial_pos=torch.tensor([0., self.settings.initial_y, 0.], device=self.adapter),
                                                   linear_velocity=torch.tensor([0., self.settings.velocity_y, 0.], device=self.adapter))

        # default knife type
        if self.settings.knife_type.lower() == "ybj":
            self.knife = Knife(KnifeType.YBJ)
        elif self.settings.knife_type.lower() == "edc":
            self.knife = Knife(KnifeType.EDC)
        elif self.settings.knife_type.lower() == "slicing":
            self.knife = Knife(KnifeType.SLICING)
        else:
            raise TypeError(
                f"Unknown knife type \"{self.settings.knife_type}\" provided in settings")
        self.knife_faces = None
        self.knife_vertices = None

        # here is the first time we import dFlex, which is where the kernels may get rebuilt if needed
        import dflex as df
        from dflex.sim import ModelBuilder
        self.builder = ModelBuilder()

        self.show_cutting_surface = show_cutting_surface

        self.progress_bar_fn = lambda *args: tqdm(
            *args, desc=self.experiment_name)
        self.requires_grad = requires_grad
        if not requires_grad:
            self.disable_gradients()

        self.settings["mu"], self.settings["lambda"] = convert_lame(
            young=settings.young, poisson=settings.poisson)

        print(
            f"Converted Young's modulus {settings.young} and Poisson's ratio {settings.poisson} to Lame parameters mu = {self.settings['mu']} and lambda = {self.settings['lambda']}"
        )

        object_rotation = df.quat_from_axis_angle(
            settings.geometry.rotation[:3], settings.geometry.rotation[3])
        if settings.generator in ("ansys", "meshio"):
            if settings.generator == "ansys":
                indices, vertices = load_ansys_mesh(
                    settings.generators.ansys.filename)
            else:
                indices, vertices = load_mesh(
                    settings.generators.meshio.filename)

            print(
                f"Loaded mesh with {np.shape(vertices)[0]} vertices and {len(indices)//4} tets.")

            # center mesh along z
            size_z = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
            vertices[:, 2] += -np.min(vertices[:, 2]) - size_z / 2.0
            self.builder.add_soft_mesh(pos=settings.geometry.position,
                                       rot=object_rotation,
                                       scale=settings.geometry.scale,
                                       vel=(0.0, 0.0, 0.0),
                                       vertices=vertices,
                                       indices=indices,
                                       density=settings.density,
                                       k_mu=self.settings["mu"],
                                       k_lambda=self.settings["lambda"],
                                       k_damp=settings.damping)
        elif settings.generator == "grid":
            self.builder.add_soft_grid(pos=settings.geometry.position,
                                       rot=object_rotation,
                                       vel=(0.0, 0.0, 0.0),
                                       dim_x=settings.generators.grid.dim_x,
                                       dim_y=settings.generators.grid.dim_y,
                                       dim_z=settings.generators.grid.dim_z,
                                       cell_x=settings.generators.grid.cell_x,
                                       cell_y=settings.generators.grid.cell_y,
                                       cell_z=settings.generators.grid.cell_z,
                                       density=settings.density,
                                       k_mu=self.settings["mu"],
                                       k_lambda=self.settings["lambda"],
                                       k_damp=settings.damping)
        else:
            print("Unknown generator \"%s\" in settings." % settings.generator)
            sys.exit(1)

        if self.settings.integrator == "explicit":
            self.integrator = df.sim.SemiImplicitIntegrator()
        elif self.settings.integrator == "implicit":
            raise NotImplementedError(
                "Implicit integration not implemented yet")
        else:
            assert False, f"Unknown integrator \"{self.settings.integrator}\" selected."

        self.hist_time = []
        self.hist_knife_force = None
        self.hist_knife_pos = None
        self.hist_knife_rot = None
        self.hist_knife_vel = None
        self.hist_cut_stiffness_min = []
        self.hist_cut_stiffness_max = []
        self.hist_cut_stiffness_mean = []

        self.groundtruth = None
        self.groundtruth_torch = None

        self.logger = None
        self.model = None

        self.nodes_above_cut = np.zeros(0)
        self.nodes_below_cut = np.zeros(0)
        self.com_above_cut = np.zeros(3)
        self.com_below_cut = np.zeros(3)
        masses = np.array(self.builder.particle_mass)
        xs = np.array(self.builder.particle_q)
        self.com = np.sum(xs * masses[:, None], axis=0) / np.sum(masses)
        self.np_particle_q = np.array(self.builder.particle_q)
        self.builder.sdf_radius = settings.sdf_radius

    def disable_gradients(self):
        """
        Disable gradient computation globally for the entire simulation back-end.
        """
        import dflex as df
        df.config.no_grad = True

    def setup_free_floating_knife(self, knife: Knife = None):
        """
        Set up a free-floating knife.
        """
        import dflex as df
        if knife is not None:
            self.knife = knife
        # add rigid body for the knife
        self.builder.add_articulation()
        self.knife_vertices, self.knife_faces = self.knife.create_mesh()
        self.blade_bottom_center_point_offset = torch.tensor(
            0.5 * (self.knife_vertices[4, :] + self.knife_vertices[9, :]), device=self.adapter)

        self.builder.knife_tri_indices = self.knife_faces
        self.builder.knife_tri_vertices = self.knife_vertices

        mesh = df.sim.Mesh(self.knife_vertices, self.knife_faces)
        rigid_rotation = df.quat_from_axis_angle(
            self.settings.knife_motion.rotation[:3], self.settings.knife_motion.rotation[3])
        # create XYZ prismatic joints and spherical joint as floating base
        rigid = self.builder.add_link(parent=-1, X_pj=df.transform((0.0, 0.0, 0.0),
                                      df.quat_identity()), axis=(1.0, 0.0, 0.0), type=df.JOINT_PRISMATIC, armature=0.0)
        rigid = self.builder.add_link(parent=rigid, X_pj=df.transform(
            (0.0, 0.0, 0.0), df.quat_identity()), axis=(0.0, 1.0, 0.0), type=df.JOINT_PRISMATIC, armature=0.0)
        rigid = self.builder.add_link(parent=rigid, X_pj=df.transform(
            (0.0, 0.0, 0.0), df.quat_identity()), axis=(0.0, 0.0, 1.0), type=df.JOINT_PRISMATIC, armature=0.0)
        rigid = self.builder.add_link(parent=rigid, X_pj=df.transform(
            (0.0, 0.0, 0.0), df.quat_identity()), axis=(0.0, 0.0, 0.0), type=df.JOINT_BALL, armature=0.0)
        self.builder.knife_link_index = rigid

        # the knife is a kinematic object, with masses set to zero it will not have any dynamical behavior
        knife_size = np.ones(3)
        # XXX density must be nonzero to avoid problems in articulated rigid body dynamics
        shape = self.builder.add_shape_mesh(
            rigid, mesh=mesh, scale=knife_size, density=1e-3, ke=0., kd=0., kf=0., mu=0.)

        self.builder.joint_q = np.concatenate(
            (self.settings.knife_motion.position, rigid_rotation)).tolist()
        self.builder.joint_qd = np.concatenate(
            (self.settings.knife_motion.velocity, self.settings.knife_motion.omega)).tolist()

    def setup_robotic_knife(self,
                            urdf_filename: str,
                            knife_link_index: int = -1,
                            is_floating_base: bool = False,
                            robot_base_transform=(
                                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
                            custom_knife_faces=None,
                            custom_knife_vertices=None):
        """
        Set up a fixed-base knife which is attached to a robot. The robot is loaded from a URDF file, the knife is assumed to be the collision mesh at the given link index.
        The pose of the robot defined by the given base transform.
        In case the collision mesh defined in the URDF is not adequate, a custom triangular knife mesh can be provided via the triangular face indices and mesh vertices.
        Note that this feature is not fully tested and may not work properly.
        """
        import dflex as df
        last_link_id = max(self.builder.shape_body) if len(
            self.builder.shape_body) > 0 else 0
        load_urdf(self.builder, urdf_filename, df.transform(
            *robot_base_transform), floating=is_floating_base)
        num_added_links = max(self.builder.shape_body) - last_link_id + 1
        knife_link_index = last_link_id + \
            (num_added_links + knife_link_index) % num_added_links
        self.builder.knife_link_index = knife_link_index
        if custom_knife_faces is not None and custom_knife_vertices is not None:
            self.knife_vertices = custom_knife_vertices
            self.knife_faces = custom_knife_faces
        else:
            shape_body = np.array(self.builder.shape_body)
            knife_geo_index = np.where(shape_body == knife_link_index)[0]
            assert len(
                knife_geo_index) == 1, "Only a single mesh must be attached at the link index of the knife if no custom knife mesh is given."
            knife_geo_index = knife_geo_index[0]
            assert self.builder.shape_geo_type[knife_geo_index] == df.GEO_MESH, "The specified knife link is not a mesh"
            knife_mesh = self.builder.shape_geo_src[knife_geo_index]
            self.knife_faces = np.array(knife_mesh.indices)
            self.knife_vertices = np.array(knife_mesh.vertices)
        self.builder.knife_tri_indices = self.knife_faces
        self.builder.knife_tri_vertices = self.knife_vertices

    def get_trace_cut_surface(self,
                              knife_motion: Motion = None,
                              num_time_steps=50,
                              show_cutting_surface=None,
                              blade_edge_i=4,
                              blade_edge_j=9,
                              blade_edge_scaling_last_step=10.0,
                              time_extend_factor=0.0):
        """
        Traces the motion of the knife blade over time [0, self.sim_duration] with the given number of steps and knife motion to create a triangular cutting surface.
        Here, the knife blade is defined as the edge between vertices i and j of the triangular surface mesh of the knife.
        At the last step of the knife motion, the knife blade is scaled by the given factor `blade_edge_scaling_last_step`.
        Optionally, the time domain over which the knife motion is traced can be extended by the `time_extend_factor` that prepends and appends time relative to the overall `sim_duration`.
        Returns the list of vertices, the list of face indices, and the triangle point tuples of the cutting surface.
        """
        import dflex as df
        if knife_motion is None:
            knife_motion = self.motion
        if self.knife_faces is None or self.knife_vertices is None:
            print("Creating free-floating knife")
            self.setup_free_floating_knife()
            # raise ValueError("The knife has not been set, make sure to call setup_free_floating_knife() or setup_robotic_knife() before cutting.")
        # trace knife path first
        # optionally, start earlier and end later to ensure the cut goes all the way through the mesh
        ts = torch.linspace(-time_extend_factor * self.sim_duration, self.sim_duration *
                            (1. + time_extend_factor), num_time_steps, device=self.adapter)

        rx = self.motion.linear_position(
            ts[0], self.sim_dt).detach().cpu().numpy()
        rr = self.motion.angular_position(
            ts[0], self.sim_dt).detach().cpu().numpy()
        # integrate linear velocity to get more accurate linear motion
        coarse_dt = self.sim_duration * \
            (1. + 2. * time_extend_factor) / num_time_steps
        pos = []
        rot = []
        for t in ts:
            pos.append(rx.copy())
            rot.append(df.quat_to_matrix(rr).copy())
            lin_vel = self.motion.linear_velocity(
                t, self.sim_dt).detach().cpu().numpy()
            ang_vel = self.motion.angular_velocity(
                t, self.sim_dt).detach().cpu().numpy()
            rx += lin_vel * coarse_dt  # might have to integrate at finer resolution
            drdt = df.quat_multiply(np.concatenate((ang_vel, [0.0])), rr) * 0.5
            rr += drdt * coarse_dt
            rr = rr / np.linalg.norm(rr)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_title('Trace knife path')
        # np_pos = np.array(pos)
        # ax.plot(np_pos[:,0], np_pos[:, 1], np_pos[:, 2])
        # plt.show()

        self.knife_positions = np.array(pos)

        cut_surface_vertices = []
        cut_surface_indices = []
        cut_surface_triangles = []
        kv_i = np.copy(self.knife_vertices[blade_edge_i, :])
        kv_j = np.copy(self.knife_vertices[blade_edge_j, :])
        for t in range(len(ts)):
            # compute knife blade line vertices
            if t == len(ts) - 1:
                # make knife much bigger at the last time step to ensure the cut is complete
                kv_i[1] *= blade_edge_scaling_last_step
                kv_j[1] *= blade_edge_scaling_last_step
            v1 = (rot[t] @ kv_i.T).T + pos[t]
            v2 = (rot[t] @ kv_j.T).T + pos[t]
            cut_surface_vertices.append(v1)
            cut_surface_vertices.append(v2)
            if t > 0:
                tri1 = (2 * t - 2, 2 * t - 1, 2 * t)
                tri2 = (2 * t, 2 * t - 1, 2 * t + 1)
                cut_surface_indices.append(tri1)
                cut_surface_indices.append(tri2)
                cut_surface_triangles.append(
                    [cut_surface_vertices[j] for j in tri1])
                cut_surface_triangles.append(
                    [cut_surface_vertices[j] for j in tri2])

        self.cut_surface_vertices = np.array(cut_surface_vertices)
        self.cut_surface_indices = np.array(cut_surface_indices)
        self.cut_surface_triangles = np.array(cut_surface_triangles)

        if show_cutting_surface is None:
            show_cutting_surface = self.show_cutting_surface
        if show_cutting_surface:
            # builder.add_cloth_grid(vel=(0, 0.0, 0.0), mass=0, **grid_args)
            cutting_surface = self.builder.add_rigid_body()
            cutting_surface_mesh = df.sim.Mesh(
                cut_surface_vertices, np.hstack(cut_surface_indices))
            self.builder.add_shape_mesh(
                cutting_surface, mesh=cutting_surface_mesh, density=0.0, ke=0., kd=0., kf=0., mu=0.)

        return self.cut_surface_vertices, self.cut_surface_indices, self.cut_surface_triangles

    def prepare_cut(self, cut_surface_triangles=None):
        if cut_surface_triangles is None:
            cut_surface_triangles = self.cut_surface_triangles
        self.builder.prepare_cut(
            self.builder.tet_indices,
            cut_surface_triangles,
            cut_spring_ke=self.settings.cut_spring_ke,
            cut_spring_kd=self.settings.cut_spring_kd,
            cut_spring_softness=self.settings.cut_spring_softness,
            cut_spring_rest_length=self.settings.cut_spring_rest_length,
            surface_cut_spring_ke=self.settings.surface_cut_spring_ke,
            surface_cut_spring_kd=self.settings.surface_cut_spring_kd,
            surface_cut_spring_softness=self.settings.surface_cut_spring_softness,
            surface_cut_spring_rest_length=self.settings.surface_cut_spring_rest_length,
            contact_ke=self.settings.sdf_ke,
            contact_kd=self.settings.sdf_kd,
            contact_kf=self.settings.sdf_kf,
            contact_mu=self.settings.sdf_mu,
            surface_contact_ke=self.settings.surface_sdf_ke,
            surface_contact_kd=self.settings.surface_sdf_kd,
            surface_contact_kf=self.settings.surface_sdf_kf,
            surface_contact_mu=self.settings.surface_sdf_mu,
        )

    def get_nodes_above_and_below_cut(self, cut_surface_triangles, cut_duplicated_x=None, vertices=None):
        if cut_duplicated_x is None:
            cut_duplicated_x = self.builder.cut_duplicated_x
        if vertices is None:
            vertices = self.builder.particle_q
        cut_surface_triangle_normals = np.array(
            [np.cross(p2 - p1, p3 - p1) for (p1, p2, p3) in cut_surface_triangles])
        cut_surface_triangle_centroids = np.mean(cut_surface_triangles, axis=1)

        centroid_tree = KDTree(cut_surface_triangle_centroids)

        # determine vertices above/below cutting surface
        nodes_above_cut = set()
        nodes_below_cut = set()
        duplicated = {duplication: original for (
            original, duplication) in cut_duplicated_x.items()}
        for i, x in enumerate(vertices):
            if i in duplicated:
                if duplicated[i] in nodes_above_cut:
                    nodes_below_cut.add(i)
                elif duplicated[i] in nodes_below_cut:
                    nodes_above_cut.add(i)
                else:
                    raise RuntimeWarning(
                        "Could not determine whether a duplicated node is above or below the cut")
            else:
                tri_id = centroid_tree.query(x)[1]
                centroid = cut_surface_triangle_centroids[tri_id]
                if np.dot(cut_surface_triangle_normals[tri_id], x - centroid) > 0:
                    nodes_above_cut.add(i)
                else:
                    nodes_below_cut.add(i)
        nodes_above_cut = np.array(list(nodes_above_cut))
        nodes_below_cut = np.array(list(nodes_below_cut))
        return nodes_above_cut, nodes_below_cut

    def cut(self, mode=None):
        """
        Perform the topological cutting operation where the mesh is processed given the currently selected knife and its motion.
        """
        import dflex as df

        if mode is not None:
            self.settings.cutting.mode = mode
        else:
            mode = self.settings.cutting.mode
        assert mode in {'rigid', 'hybrid', 'fem'}

        original_cut_surface_vertices, cut_surface_indices, original_cut_surface_triangles = self.get_trace_cut_surface()
        if mode is None or not self.settings.cutting.active:
            print("Cutting is not active or no cutting mode has been selected.")
            return

        if mode != "hybrid":
            self.prepare_cut()
            self.nodes_above_cut, self.nodes_below_cut = self.get_nodes_above_and_below_cut(
                self.cut_surface_triangles)

        xs = np.array(self.builder.particle_q)

        if mode != "hybrid":
            masses = np.array(self.builder.particle_mass)
            self.mass_above_cut = np.sum(masses[self.nodes_above_cut])
            self.mass_below_cut = np.sum(masses[self.nodes_below_cut])
            # self.com_above_cut = np.sum(xs[self.nodes_above_cut] * masses[self.nodes_above_cut,None], axis=0) / self.mass_above_cut
            # self.com_below_cut = np.sum(xs[self.nodes_below_cut] * masses[self.nodes_below_cut,None], axis=0) / self.mass_below_cut
            self.com_above_cut = np.mean(xs[self.nodes_above_cut], axis=0)
            self.com_below_cut = np.mean(xs[self.nodes_below_cut], axis=0)

        if mode.lower() == "fem":
            for i, x in enumerate(self.builder.particle_q):
                excluded = False
                for exclude in self.settings.static_vertices.exclude:
                    if inside(exclude, x):
                        excluded = True
                        continue
                if excluded:
                    continue
                for include in self.settings.static_vertices.include:
                    if inside(include, x):
                        self.builder.particle_mass[i] = 0

            self.create_model_()
            return

        raise NotImplementedError(
            "Cutting modes other than \"fem\" are not supported at the moment")

    def create_model_(self):
        """
        Sets up the simulation model that stores the static information used by the simulator.
        """
        self.builder.sim_duration = self.sim_duration

        self.model = self.builder.finalize(adapter=self.adapter,
                                           knife=self.knife,
                                           minimum_mass=self.settings.minimum_particle_mass,
                                           requires_grad=self.requires_grad)

        # generate contact points (only if the mesh parts are simulated via rigid bodies)
        # self.model.collide(None)

        # disable triangle dynamics (just used for rendering)
        self.model.tri_ke = 0.0
        self.model.tri_ka = 0.0
        self.model.tri_kd = 0.0
        self.model.tri_kb = 0.0

        self.model.contact_ke = self.settings.ground_ke
        self.model.contact_kd = self.settings.ground_kd
        self.model.contact_kf = self.settings.ground_kf
        self.model.contact_mu = self.settings.ground_mu
        self.model.contact_closeness = self.settings.ground_radius

        self.model.particle_radius = 0.005
        self.model.ground = self.settings.ground

        self.model.nodes_above_cut = self.nodes_above_cut
        self.model.nodes_below_cut = self.nodes_below_cut
        self.model.com = torch.tensor(
            self.com, device=self.adapter, dtype=torch.float32)
        self.model.com_above_cut = torch.tensor(
            self.com_above_cut, device=self.adapter, dtype=torch.float32)
        self.model.com_below_cut = torch.tensor(
            self.com_below_cut, device=self.adapter, dtype=torch.float32)
        # self.model.rigid_cutting = self.rigid_cutting

        self.model.initial_x = torch.tensor(
            self.builder.particle_q, device=self.adapter, dtype=torch.float32)
        # self.model.initial_x[self.nodes_above_cut] -= self.model.com_above_cut
        # self.model.initial_x[self.nodes_below_cut] -= self.model.com_below_cut

    def assign_parameters(self):
        """
        Assigns simulation parameters that may be optimized from roll-outs of the differentiable simulator.
        """
        if "initial_y" in self.parameters:
            assert type(
                self.motion) == ConstantLinearVelocityMotion, "Knife motion must be of type ConstantLinearVelocityMotion to infer 'initial_y' parameter."
            self.motion.initial_pos[1] = self.parameters["initial_y"].assignable_tensor(
            )
        if "velocity_y" in self.parameters:
            assert type(
                self.motion) == ConstantLinearVelocityMotion, "Knife motion must be of type ConstantLinearVelocityMotion to infer 'velocity_y' parameter."
            self.motion.lin_vel[1] = self.parameters["velocity_y"].assignable_tensor(
            )
        if "cut_spring_ke" in self.parameters:
            self.state.cut_spring_ke = self.parameters["cut_spring_ke"].assignable_tensor(
            )
            self.model.cut_spring_stiffness = self.parameters["cut_spring_ke"].assignable_tensor(
            )
        if "cut_spring_kd" in self.parameters:
            self.state.cut_spring_kd = self.parameters["cut_spring_kd"].assignable_tensor(
            )
        if "cut_spring_softness" in self.parameters:
            self.model.cut_spring_softness = self.parameters["cut_spring_softness"].assignable_tensor(
            )
        if "sdf_radius" in self.parameters:
            self.model.sdf_radius = self.parameters["sdf_radius"].assignable_tensor(
            )
        if "sdf_ke" in self.parameters:
            self.model.sdf_ke = self.parameters["sdf_ke"].assignable_tensor()
        if "sdf_kd" in self.parameters:
            self.model.sdf_kd = self.parameters["sdf_kd"].assignable_tensor()
        if "sdf_kf" in self.parameters:
            self.model.sdf_kf = self.parameters["sdf_kf"].assignable_tensor()
        if "sdf_mu" in self.parameters:
            self.model.sdf_mu = self.parameters["sdf_mu"].assignable_tensor()
        if "mu" in self.parameters:
            self.model.tet_mu = self.parameters["mu"].assignable_tensor()
        if "lambda" in self.parameters:
            self.model.tet_lambda = self.parameters["lambda"].assignable_tensor(
            )
        if "damping" in self.parameters:
            self.model.tet_damping = self.parameters["damping"].assignable_tensor(
            )

    def init_parameter_(self, param_name, src_tensor):
        """
        Initializes a simulation parameter as optimization variable from a tensor and defines its parameter bounds.
        """
        if param_name in self.parameters:
            tensor = self.parameters[param_name].create_tensor(
                src_tensor, self.adapter)
            if not self.parameters[param_name].fixed:
                self.optim_params.append({
                    "name": param_name,
                    "params": tensor,
                    "limit_min": self.parameters[param_name].low,
                    "limit_max": self.parameters[param_name].high,
                })
                self.parameter_names.append(param_name)

    def init_parameters(self):
        """
        Initializes the tensors for the optimizable simulation parameters.
        """
        self.parameter_names = []
        self.optim_params = []
        self.init_parameter_("velocity_y", None)
        self.init_parameter_("initial_y", None)
        self.init_parameter_("lateral_velocity", None)
        self.init_parameter_("cut_spring_ke", self.model.cut_spring_stiffness)
        self.init_parameter_("cut_spring_kd", self.model.cut_spring_damping)
        self.init_parameter_("cut_spring_softness",
                             self.model.cut_spring_softness)
        self.init_parameter_("sdf_radius", self.model.sdf_radius)
        self.init_parameter_("sdf_ke", self.model.sdf_ke)
        self.init_parameter_("sdf_kd", self.model.sdf_kd)
        self.init_parameter_("sdf_kf", self.model.sdf_kf)
        self.init_parameter_("sdf_mu", self.model.sdf_mu)
        self.init_parameter_("mu", self.model.tet_mu)
        self.init_parameter_("lambda", self.model.tet_lambda)
        self.init_parameter_("damping", self.model.tet_damping)
        print("Selected the following parameters for optimization:\n\t[%s]\n" % ", ".join(
            self.parameter_names))
        return self.optim_params

    def simulation_step(self):
        """
        Perform a single simulation step where the dynamics model is evaluated and the state is updated from the integrator.
        """
        import dflex as df

        # torch_time = torch.tensor(self.sim_time, device=self.adapter)
        # self.motion.update_state(self.state, torch_time, self.sim_dt)
        self.motion.update_state(self.state, self.sim_time, self.sim_dt)

        # forward dynamics
        with df.ScopedTimer("simulate", False):
            # simulation step
            # we do not need to update the mass matrix since we assume the motion of the knife (robot) is prescribed (kinematic, not dynamic)
            self.state = self.integrator.forward(
                self.model, self.state, self.sim_dt,
                update_mass_matrix=False)
            self.sim_time += self.sim_dt
            self.sim_step += 1

    def init_sim_structures_(self):
        """
        Initializes internal simulation structures at the beginning of a simulation.
        """
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = max(1, self.sim_substeps)
        self.sim_coarse_steps = self.sim_steps // self.sim_substeps

        self.hist_time = []
        self.hist_cut_stiffness_min = []
        self.hist_cut_stiffness_max = []
        self.hist_cut_stiffness_mean = []
        self.hist_knife_pos = torch.zeros(
            (self.sim_coarse_steps, 3), device=self.adapter)
        self.hist_knife_rot = torch.zeros(
            (self.sim_coarse_steps, 4), device=self.adapter)
        self.hist_knife_vel = torch.zeros(
            (self.sim_coarse_steps, 3), device=self.adapter)
        self.hist_knife_force = torch.zeros(
            self.sim_coarse_steps, device=self.adapter)

    def simulate(self, render=False, show_progressbar=True):
        """
        Roll out a knife force profile over the defined self.sim_duration by simulating the dynamics model.
        This function will only simulate force measurements at the coarse simulation steps defined by the number of sim_substeps that are skipped.
        """
        self.init_sim_structures_()

        del self.model
        self.create_model_()

        self.state = self.model.state()

        self.assign_parameters()

        if render:
            # set up Usd renderer
            from pxr import Usd
            from dflex.render import UsdRenderer
            stage_name = f"outputs/{self.experiment_name}.usd"
            stage = Usd.Stage.CreateNew(stage_name)
            renderer = UsdRenderer(self.model, stage)
            renderer.draw_points = False
            renderer.draw_springs = False
            renderer.draw_shapes = True

        if show_progressbar:
            step_range = self.progress_bar_fn(range(self.sim_coarse_steps))
        else:
            step_range = range(self.sim_coarse_steps)

        for step in step_range:
            for _ in range(self.sim_substeps):
                self.simulation_step()

            # render
            if render:
                renderer.update(self.state, self.sim_time)

            self.hist_time.append(self.sim_time)
            self.hist_cut_stiffness_min.append(
                torch.min(self.state.cut_spring_ke).item())
            self.hist_cut_stiffness_max.append(
                torch.max(self.state.cut_spring_ke).item())
            self.hist_cut_stiffness_mean.append(
                torch.mean(self.state.cut_spring_ke).item())

            knife_f = torch.sum(torch.norm(self.state.knife_f, dim=1))
            assert (not np.isnan(knife_f.item(
            ))), f"Knife force is NaN at step {step}/{self.sim_coarse_steps} (time: {self.sim_time:.3f}s)"

            self.hist_knife_force[step] = self.hist_knife_force[step] + knife_f
            knife_x = self.state.body_X_sm[self.model.knife_link_index]
            knife_v = self.state.body_v_s[self.model.knife_link_index]
            self.hist_knife_pos[step] = self.hist_knife_pos[step] + knife_x[:3]
            self.hist_knife_rot[step] = self.hist_knife_rot[step] + knife_x[3:]
            self.hist_knife_vel[step] = self.hist_knife_vel[step] + knife_v[:3]

        with torch.no_grad():
            if torch.sum(self.hist_knife_force).item() < 1e-4:
                print(colored(
                    f"WARNING: Predicted knife force history is zero in experiment {self.experiment_name}!", fg.red), file=sys.stderr)

        if render:
            stage.Save()
            print(f"Saved USD stage at {stage_name}.")

        del self.state
        self.state = None

        # return loss
        return self.hist_knife_force

    def get_cutting_spring_midpoints(self, xs=None):
        """
        Computes center points of the cutting springs.
        """
        if len(self.builder.cut_edge_indices) == 0:
            return np.zeros((0, 3))
        if xs is None:
            xs = self.builder.particle_q
        elif type(xs) == torch.Tensor:
            xs = xs.detach().cpu().numpy()
        ids = np.array(self.builder.cut_edge_indices)
        coords = np.array(self.builder.cut_edge_coords)
        spring_ids = np.array(self.builder.cut_spring_indices)
        tri_points = xs[ids[:, 0]] * \
            (1. - coords[-1, None]) + xs[ids[:, 1]] * (coords[-1, None])
        aug_xs = np.vstack((xs, tri_points))
        l1s = ids[spring_ids[:, 0], 0]
        l2s = ids[spring_ids[:, 0], 1]
        # note swapped coordinates below the cut
        r1s = ids[spring_ids[:, 1], 1]
        r2s = ids[spring_ids[:, 1], 0]
        lcs = coords[spring_ids[:, 0]]
        rcs = coords[spring_ids[:, 1]]
        xis = aug_xs[l1s] * (1 - lcs[:, None]) + aug_xs[l2s] * lcs[:, None]
        xjs = aug_xs[r1s] * rcs[:, None] + aug_xs[r2s] * (1 - rcs[:, None])
        center = 0.5 * (xis + xjs)
        return center

    def visualize_cut(self,
                      use_latex_colors=False,
                      show_knife_trace_mesh=True,
                      show_node_side=False,
                      cut_separation=0.01,
                      plotter=None,
                      app=None,
                      execute_blocking_io=False):
        """
        Visualizes the mesh and cutting springs that are inserted as part of the cutting operation.
        """
        from PyQt5 import Qt
        import pyvista as pv

        def convert_viz_indices(faces):
            # bring N indices into pyvista format [N, i, j, k]
            return np.hstack(np.hstack([[[faces.shape[1]]] * faces.shape[0], faces]))

        if app is None:
            app = Qt.QApplication(sys.argv)

        if plotter is None:
            from pyvistaqt import BackgroundPlotter
            plotter = BackgroundPlotter(
                title='Cut Visualization', auto_update=20.)
            plotter.show()

        # plotter.add_axes_at_origin(labels_off=True)
        plotter.set_viewup([0., 1., 0.])
        plotter.set_focus([0., 0., 0.])
        plotter.set_position([0.0, .2, -0.01], True)
        if use_latex_colors:
            plotter.set_background("white")

        mc = self.builder

        if show_knife_trace_mesh:
            cutting_mesh = pv.PolyData(self.cut_surface_vertices, np.array(
                [(3, *i) for i in self.cut_surface_indices]))
            plotter.add_mesh(cutting_mesh, style="wireframe", show_edges=True,
                             render_lines_as_tubes=True, line_width=2., color='red', opacity=0.5)

        cut_tri_indices = np.array(mc.cut_tri_indices)
        cut_virtual_tri_indices_above_cut = np.array(
            mc.cut_virtual_tri_indices_above_cut)
        cut_virtual_tri_indices_below_cut = np.array(
            mc.cut_virtual_tri_indices_below_cut)
        ps = np.array(mc.particle_q).copy()
        if cut_separation > 0.0:
            # compute separation axis
            sep_axis = self.com_above_cut - self.com_below_cut
            sep_axis_norm = np.linalg.norm(sep_axis)
            if sep_axis_norm > 0.0:
                sep_axis /= sep_axis_norm
            sep_axis *= cut_separation
            if len(self.nodes_above_cut) > 0:
                ps[self.nodes_above_cut] += sep_axis
            if len(self.nodes_below_cut) > 0:
                ps[self.nodes_below_cut] -= sep_axis
        cut_edge_coords = np.array(mc.cut_edge_coords)
        cut_edge_indices = np.array(mc.cut_edge_indices)
        cut_virtual_tri_indices = np.array(mc.cut_virtual_tri_indices)

        center = self.get_cutting_spring_midpoints(ps)
        if len(center) > 0:
            plotter.add_mesh(pv.PolyData(center), point_size=8,
                             render_points_as_spheres=True, color="green")

        cut_tets = set(mc.cut_tets)
        non_cut_tets = [tet for i, tet in enumerate(
            mc.tet_indices) if i not in cut_tets]
        top = MeshTopology(non_cut_tets)
        unique_edges_mesh = pv.PolyData(ps, convert_viz_indices(
            np.array(list(top.unique_edges.keys()))))
        plotter.add_mesh(unique_edges_mesh, style="wireframe", render_lines_as_tubes=True,
                         line_width=1.5, color='gray' if use_latex_colors else 'cyan')

        ids = np.array(self.builder.cut_edge_indices)
        if len(ids) > 0:
            coords = np.array(self.builder.cut_edge_coords)
            spring_ids = np.array(self.builder.cut_spring_indices)
            tri_points = ps[ids[:, 0]] * \
                (1. - coords[-1, None]) + ps[ids[:, 1]] * (coords[-1, None])
            aug_xs = np.vstack((ps, tri_points))
            l1s = ids[spring_ids[:, 0], 0]
            l2s = ids[spring_ids[:, 0], 1]
            # note swapped coordinates below the cut
            r1s = ids[spring_ids[:, 1], 1]
            r2s = ids[spring_ids[:, 1], 0]
            lcs = coords[spring_ids[:, 0]]
            rcs = coords[spring_ids[:, 1]]
            xis = aug_xs[l1s] * (1 - lcs[:, None]) + aug_xs[l2s] * lcs[:, None]
            xjs = aug_xs[r1s] * rcs[:, None] + aug_xs[r2s] * (1 - rcs[:, None])
            plotter.add_mesh(pv.PolyData(xis), point_size=10,
                             render_points_as_spheres=True, color="white")
            plotter.add_mesh(pv.PolyData(xjs), point_size=10,
                             render_points_as_spheres=True, color="pink")

        if len(cut_virtual_tri_indices) > 0:
            cut_virtual_tri_indices = convert_viz_indices(
                cut_virtual_tri_indices)

            cut_vertices = ps[cut_edge_indices[:, 0]] * (1.0 - cut_edge_coords)[:, None] + \
                ps[cut_edge_indices[:, 1]] * cut_edge_coords[:, None]
            # cut_vertices = ps[cut_edge_indices[:, 0]] * cut_edge_coords[:, None] + \
            #     ps[cut_edge_indices[:, 1]] * (1.0 - cut_edge_coords)[:, None]

            cut_virtual_tri_indices_above_cut = convert_viz_indices(
                cut_virtual_tri_indices_above_cut)
            cut_virtual_tris_above_cut = pv.PolyData(
                cut_vertices, cut_virtual_tri_indices_above_cut)
            plotter.add_mesh(cut_virtual_tris_above_cut, show_edges=True, edge_color='magenta',
                             color='magenta', line_width=2, render_lines_as_tubes=False)
            cut_virtual_tri_indices_below_cut = convert_viz_indices(
                cut_virtual_tri_indices_below_cut)
            cut_virtual_tris_below_cut = pv.PolyData(
                cut_vertices, cut_virtual_tri_indices_below_cut)
            plotter.add_mesh(cut_virtual_tris_below_cut,
                             show_edges=True,
                             edge_color='yellow',
                             color='yellow',
                             line_width=2,
                             render_lines_as_tubes=False,
                             opacity=1.0)

        # plotter.add_mesh(pv.PolyData(ps), point_size=10, render_points_as_spheres=True, color="pink")
        if show_node_side:
            above_cut = ps[self.nodes_above_cut]
            below_cut = ps[self.nodes_below_cut]
            plotter.add_mesh(pv.PolyData(above_cut), point_size=10,
                             render_points_as_spheres=True, color="pink")
            plotter.add_mesh(pv.PolyData(below_cut), point_size=10,
                             render_points_as_spheres=True, color="white")

        if execute_blocking_io:
            sys.exit(app.exec_())

        return app

    def __plot_knife_motion(self, ax0=0, ax1=1, num_steps_to_plot=20):
        """
        Helper function to plot the knife motion.
        """
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import dflex as df

        def plot_knife_outline(pos, rot, ax0, ax1, alpha=0.8):
            knife_vertices, _ = self.knife.create_mesh()
            xs = np.array([df.quat_rotate(rot, x) +
                          pos for x in knife_vertices])
            lines = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 5), (1, 6), (5, 6), (2, 7),
                     (3, 8), (5, 7), (5, 6), (6, 8), (7, 9), (8, 9), (2, 4), (3, 4), (4, 9)]
            for i, j in lines:
                plt.plot([xs[i, ax0], xs[j, ax0]], [xs[i, ax1], xs[j, ax1]],
                         c='k', linewidth=1, alpha=alpha, zorder=3)
            return xs

        np_hist_knife_pos = self.hist_knife_pos.detach().cpu().numpy()
        np_hist_knife_rot = self.hist_knife_rot.detach().cpu().numpy()

        plt.title("Knife Position")
        plt.grid(True)

        # plot convex hull of the mesh
        hull = ConvexHull(self.np_particle_q[:, [ax0, ax1]])
        hull = self.np_particle_q[hull.vertices]
        plt.scatter(hull[:, ax0], hull[:, ax1], color='k', s=3)
        plt.fill(hull[:, ax0], hull[:, ax1], 'brown', alpha=0.3, zorder=3)

        # plot trace of the knife
        plt.plot(np_hist_knife_pos[:, ax0],
                 np_hist_knife_pos[:, ax1], alpha=0.0)
        colorline(
            plt.gca(), np_hist_knife_pos[:, ax0], np_hist_knife_pos[:, ax1], zorder=4)
        num_steps_to_plot = min(
            num_steps_to_plot, self.hist_knife_pos.shape[0])
        alpha_factor = 0.6 / num_steps_to_plot
        i_factor = len(np_hist_knife_pos) // num_steps_to_plot
        for i in range(num_steps_to_plot):
            alpha = (i + 1) * alpha_factor
            ii = i * i_factor
            knife_xs = plot_knife_outline(
                np_hist_knife_pos[ii], np_hist_knife_rot[ii], ax0, ax1, alpha=alpha)
            try:
                hull = ConvexHull(knife_xs[:, [ax0, ax1]])
                hull = knife_xs[hull.vertices]
                polygon = Polygon(hull[:, [ax0, ax1]], True)
                p = PatchCollection([polygon], alpha=alpha, facecolors=['C0'], edgecolor=[
                                    'black'], linewidths=(1, ), zorder=2)
                plt.gca().add_collection(p)
            except:
                pass
        xyz = 'xyz'
        plt.xlabel(xyz[ax0])
        plt.ylabel(xyz[ax1])
        plt.axis("equal")
        if self.settings.ground:
            plt.axhline(0.0, color='k', alpha=0.5)

    def plot_simulation_results(self):
        """
        Generates a 2x2 summary plot of the simulation results.
        """
        np_hist_knife_pos = self.hist_knife_pos.detach().cpu().numpy()
        np_hist_knife_force = self.hist_knife_force.detach().cpu().numpy()
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.clf()
        plt.subplot(221)
        plt.title("Cut Spring Stiffness")
        plt.fill_between(self.hist_time, self.hist_cut_stiffness_min,
                         self.hist_cut_stiffness_max, color="C0", alpha=0.5)
        plt.plot(self.hist_time, self.hist_cut_stiffness_mean, color="C0")
        plt.grid()
        plt.subplot(223)
        self.__plot_knife_motion(ax0=2, ax1=1)
        # plt.title("Knife position")
        # plt.plot(np_hist_knife_pos[:, 2], np_hist_knife_pos[:, 1], alpha=0.0)
        # colorline(plt.gca(), np_hist_knife_pos[:, 2], np_hist_knife_pos[:, 1])
        # plt.xlabel("z")
        # plt.ylabel("y")
        # plt.grid()
        plt.subplot(222)
        plt.title("Knife Force")
        if self.groundtruth is not None:
            plt.plot(np.linspace(0., self.sim_coarse_dt * len(self.groundtruth),
                     len(self.groundtruth)), self.groundtruth, color="C2", label="Groundtruth")
        plt.plot(np.linspace(0., self.sim_time, len(np_hist_knife_force)),
                 np_hist_knife_force, color="C3", label="Predicted")
        if self.groundtruth is not None:
            plt.legend()
        plt.grid()
        plt.subplot(224)
        self.__plot_knife_motion(ax0=0, ax1=1)
        plt.tight_layout()
        return fig

    def load_groundtruth(self, groundtruth, groundtruth_dt=None):
        """
        Load groundtruth force profile. If `groundtruth` is a string, it will be loaded from the given path.
        Otherwise, it is assumed to be an array of the knife forces at each time step.
        """
        if type(groundtruth) == str:
            if groundtruth.endswith(".npy"):
                self.groundtruth = np.load(groundtruth)
                if groundtruth_dt is not None and groundtruth_dt != self.sim_coarse_dt:
                    print(
                        f"Resampling groundtruth from dt = {groundtruth_dt} to {self.sim_coarse_dt}")
                    from scipy import signal
                    ratio = groundtruth_dt / self.sim_coarse_dt
                    self.groundtruth = signal.resample_poly(
                        self.groundtruth, int(10 * ratio), 10)
            elif groundtruth.endswith(".pkl"):
                log = pickle.load(open(groundtruth, "rb"))
                self.groundtruth = log["hist_knife_force"]
                groundtruth_dt = log["settings"]["sim_dt"]
                if groundtruth_dt != self.sim_coarse_dt:
                    print(
                        f"Resampling groundtruth from dt = {groundtruth_dt} to {self.sim_coarse_dt}")
                    from scipy import signal
                    ratio = groundtruth_dt / self.sim_coarse_dt
                    self.groundtruth = signal.resample_poly(
                        self.groundtruth, int(10 * ratio), 10)
            elif groundtruth.endswith("resultant_force_xyz.csv"):
                from scipy import signal
                # ANSYS force file
                data = pd.read_csv(groundtruth, header=1)
                E = self.settings.young
                rho = self.settings.density
                omega = np.sqrt(E / rho) / 100.
                b, a = signal.butter(N=50, Wn=omega)
                signals = []
                # plt.subplot(121)
                # plt.grid()
                # get this weird number in the column title
                column_mid_id = data.axes[1][3].split()[1]
                for c in ["X", "Y", "Z"]:
                    filtered = signal.lfilter(
                        b, a, data[f"Ma {column_mid_id} {c}-force"])
                    # plt.plot(data["Time"], filtered, label=c)
                    signals.append(filtered)
                # plt.legend()
                # plt.subplot(122)
                norms = np.linalg.norm(signals, axis=0)
                # plt.grid()
                # plt.plot(data["Time"], norms)
                interpolator = interp1d(data["Time"], norms)
                self.groundtruth = interpolator(
                    np.arange(0.0, data["Time"].values[-1], self.sim_coarse_dt))
                # plt.show()
                # plt.plot(self.groundtruth)
                # plt.show()
            else:
                print("Groundtruth must be either a npy or a pkl file!",
                      file=sys.stderr)
                exit(1)
        elif type(groundtruth) in (np.ndarray, list):
            self.groundtruth = groundtruth
        else:
            print("Groundtruth must be either a string or an array!", file=sys.stderr)
            exit(1)

        self.groundtruth_torch = torch.tensor(
            self.groundtruth[:self.sim_steps], dtype=torch.float32, device=self.adapter)

    def visualize(self, skip_steps=10, **kwargs):
        """
        Creates an interactive visualizer and runs the simulation.
        """
        from cutting import Visualizer
        from PyQt5 import Qt
        app = Qt.QApplication(sys.argv)
        _ = Visualizer(self, skip_steps=skip_steps, **kwargs)
        sys.exit(app.exec_())

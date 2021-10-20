
# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Optimize the knife slicing motion via the Modified Differential Method of Multipliers (MDMM),
which minimizes the total force exerted by the knife, while ensuring that the knife blade
touches the ground at the end of the cut and that it does not slide laterally further than its
blade length.
This code will log plots and scalars of the current optimization progress to tensorboard.
Run the following command in the terminal to launch tensorboard:
    tensorboard --logdir=log
"""

# fmt: off
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from datetime import datetime

import tqdm
import torch

from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cutting import load_settings, SlicingMotion
from cutting import CuttingSim

import mdmm
# fmt: on

sns.set_palette("tab10")

settings = load_settings("examples/config/ansys_sphere_apple.json")
settings.sim_duration = 0.6
settings.sim_dt = 2e-5
settings.initial_y = 0.08
slicing_kernel_width = 0.02
slicing_waypoints = 5
device = "cuda"
now = datetime.now()
experiment_name = f"optimize_slicing_wp{slicing_waypoints}_dt{settings.sim_dt}_{now.strftime('%Y%m%d-%H%M')}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")

requires_grad = True

sim = CuttingSim(settings, experiment_name=experiment_name,
                 adapter=device, requires_grad=requires_grad)

slicing_amplitudes = torch.tensor(np.ones(
    slicing_waypoints) * 0.02, device=device, dtype=torch.float32, requires_grad=requires_grad)
pressing_velocities = torch.tensor(np.ones(
    slicing_waypoints) * -0.05, device=device, dtype=torch.float32, requires_grad=requires_grad)
slicing_frequency = torch.tensor(
    5., device=device, dtype=torch.float32, requires_grad=requires_grad)
sim.motion = SlicingMotion(initial_pos=torch.tensor([0.0, sim.settings.initial_y, 0.], device=sim.adapter),
                           slicing_frequency=slicing_frequency,
                           slicing_amplitudes=slicing_amplitudes,
                           pressing_velocities=pressing_velocities,
                           slicing_times=np.linspace(
                               0.0, settings.sim_duration, slicing_waypoints),
                           slicing_kernel_width=slicing_kernel_width)
fig = sim.motion.plot(sim_duration=settings.sim_duration)
# fig.show()
# plt.show()

blade_length = sim.knife.depth
mesh_min_z = np.array(sim.builder.particle_q)[:, 2].min()
mesh_max_z = np.array(sim.builder.particle_q)[:, 2].max()

height_constraint_scale = 10000.0
height_constraint_damping = 100.0

blade_constraint_scale = 10000.0
blade_constraint_damping = 100.0

learning_rate = 1e-2

# add constraint to have the blade touch the ground at the end of the cut
knife_height_constraint = mdmm.EqConstraint(lambda: sim.hist_knife_pos[-1, 1] + sim.blade_bottom_center_point_offset[1],
                                            0.0,
                                            scale=height_constraint_scale,
                                            damping=height_constraint_damping)
blade_min = mesh_max_z - blade_length / 2
blade_max = mesh_min_z + blade_length / 2
# add constraint to prevent the knife from moving laterally further than its blade length
blade_length_constraint = mdmm.EqConstraint(lambda: (sim.hist_knife_pos[:, 2].clamp(blade_min, blade_max) - sim.hist_knife_pos[:, 2]).sum(),
                                            0.0,
                                            scale=blade_constraint_scale,
                                            damping=blade_constraint_damping)

sim.cut()

hist_knife_force = sim.simulate()
fig = sim.plot_simulation_results()
fig.show()
plt.show()

# sim.visualize()

parameters = (slicing_frequency, slicing_amplitudes, pressing_velocities)

mdmm_module = mdmm.MDMM([knife_height_constraint, blade_length_constraint])
opt = mdmm_module.make_optimizer(parameters, lr=learning_rate)

for iteration in tqdm.tqdm(range(100)):
    # objective is to minimize mean knife force
    hist_knife_force = sim.simulate()
    mean_knife_force = torch.mean(hist_knife_force)
    mdmm_return = mdmm_module(mean_knife_force)

    with torch.no_grad():
        c_knife_height_constraint = knife_height_constraint()
        logger.add_scalar("knife_height_constraint/value",
                          c_knife_height_constraint.value.item(), iteration)
        logger.add_scalar("knife_height_constraint/inf",
                          c_knife_height_constraint.inf.item(), iteration)
        logger.add_scalar("knife_height_constraint/fn_value",
                          c_knife_height_constraint.fn_value.item(), iteration)
        logger.add_scalar("knife_height_constraint/lambda",
                          knife_height_constraint.lmbda.item(), iteration)
        if knife_height_constraint.lmbda.grad is not None:
            logger.add_scalar("knife_height_constraint/lambda_grad",
                              knife_height_constraint.lmbda.grad.item(), iteration)
        c_blade_length_constraint = blade_length_constraint()
        logger.add_scalar("blade_length_constraint/value",
                          c_blade_length_constraint.value.item(), iteration)
        logger.add_scalar("blade_length_constraint/inf",
                          c_blade_length_constraint.inf.item(), iteration)
        logger.add_scalar("blade_length_constraint/fn_value",
                          c_blade_length_constraint.fn_value.item(), iteration)
        logger.add_scalar("blade_length_constraint/lambda",
                          blade_length_constraint.lmbda.item(), iteration)
        if blade_length_constraint.lmbda.grad is not None:
            logger.add_scalar("blade_length_constraint/lambda_grad",
                              blade_length_constraint.lmbda.grad.item(), iteration)

    logger.add_scalar("slicing/slicing_frequency",
                      slicing_frequency.item(), iteration)
    logger.add_scalar("slicing/slicing_amplitudes_mean",
                      torch.mean(slicing_amplitudes).item(), iteration)
    logger.add_scalar("slicing/pressing_velocities_mean",
                      torch.mean(pressing_velocities).item(), iteration)
    #     logger.add_scalar("knife_height", sim.motion.get_params()[0][1], iteration)
    #     logger.add_scalar("knife_height_constraint", knife_height_constraint.get_constraint(), iteration)
    logger.add_scalar("mean_knife_force", mean_knife_force.item(), iteration)

    #     logger.add_hparams(
    #         {
    #             "height_constraint_scale": height_constraint_scale,
    #             "height_constraint_damping": height_constraint_damping,
    #             "slicing_waypoints": slicing_waypoints,
    #             "initial_y": sim.settings.initial_y,
    #             "sim_dt": settings.sim_dt,
    #             "sim_duration": settings.sim_duration,
    #             "learning_rate": learning_rate,
    #             "slicing_kernel_width": slicing_kernel_width,
    #         }, {
    #             "mean_knife_force": mean_knife_force.item(), "knife_height_constraint/inf": c_knife_height_constraint.inf.item()
    #         },
    #         "hparams",
    #         iteration)

    print("mean_knife_force", mean_knife_force)
    fig = sim.plot_simulation_results()
    fig.savefig(f"log/{experiment_name}/{experiment_name}_{iteration}.png")
    logger.add_figure("simulation", fig, iteration)
    opt.zero_grad()
    mdmm_return.value.backward()
    opt.step()

# hist_knife_force = sim.simulate()
# mean_knife_force = torch.mean(hist_knife_force)
# print("mean_knife_force", mean_knife_force)
# fig = sim.plot_simulation_results()
# fig.show()
# mean_knife_force.backward()

# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This example demonstrates how to leverage the differentiability of DiSECt to
infer the simulation parameters of our simulator to reduce the gap between
the simulated knife force profile and a ground-truth force profile.

The optimization progress is logged in tensorboard, which can be viewed by
running the following command:
    tensorboard --logdir=log

Note: this example currently allocates about 6 GB of GPU memory, which we aim to
reduce in further updates of our simulator.
"""

# fmt: off
import torch
import tqdm
import sys
import os
from datetime import datetime
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from cutting import CuttingSim
from cutting import load_settings, ConstantLinearVelocityMotion
# fmt: on


settings = load_settings("examples/config/ansys_sphere_apple.json")
settings.sim_duration = 0.4
# settings.sim_dt = 4e-5
settings.sim_dt = 1e-5
settings.initial_y = 0.075
settings.velocity_y = -0.05
device = "cuda"
learning_rate = 0.01

now = datetime.now()
experiment_name = f"param_inference_dt{settings.sim_dt}_{now.strftime('%Y%m%d-%H%M')}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")

requires_grad = True

sim = CuttingSim(settings, dataset='ansys',
                 experiment_name=experiment_name, adapter=device, requires_grad=True)
sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.0, settings.initial_y, 0.0], device=device),
    linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

sim.cut()

# sim.visualize()

opt_params = sim.init_parameters()

sim.load_groundtruth('dataset/forces/sphere_fine_resultant_force_xyz.csv')

opt = torch.optim.Adam(opt_params, lr=learning_rate)

for iteration in tqdm.trange(100):
    sim.motion = ConstantLinearVelocityMotion(
        initial_pos=torch.tensor(
            [0.0, settings.initial_y, 0.0], device=device),
        linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

    print(f'\n### {experiment_name}  --  Iteration {iteration}')

    hist_knife_force = sim.simulate()
    loss = torch.square(hist_knife_force -
                        sim.groundtruth_torch[:len(hist_knife_force)]).mean()
    print("Loss:", loss.item())

    for name, param in sim.parameters.items():
        logger.add_scalar(
            f"{name}/value", param.actual_tensor_value.mean().item(), iteration)

    logger.add_scalar("loss", loss.item(), iteration)

    fig = sim.plot_simulation_results()
    fig.savefig(f"log/{experiment_name}/{experiment_name}_{iteration}.png")
    logger.add_figure("simulation", fig, iteration)
    opt.zero_grad()
    loss.backward(retain_graph=False)
    for name, param in sim.parameters.items():
        if param.tensor.grad is None:
            print(
                f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad N/A!')
            print(f"Iteration {iteration}: {name} has no gradient!")
        else:
            print(
                f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad = {param.tensor.grad.mean().item()}')
            logger.add_scalar(
                f"{name}/grad", param.tensor.grad.mean().item(), iteration)

    opt.step()

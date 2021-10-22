# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This example demonstrates how to generate a USD file from the simulation.
It stores the simulation as an animation which can be played back in 3D
software that supports USD, such as Pixar's usdview or NVIDIA Omniverse.

The `sim_substeps` parameter defines how many simulation steps are skipped
between each frame that is stored in the USD file.

Once the rendering is complete, the USD file is saved at outputs/render_demo.usd
"""

# fmt: off
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from cutting import load_settings, ConstantLinearVelocityMotion, CuttingSim
# fmt: on

settings = load_settings("examples/config/ansys_cylinder_jello.json")
settings.sim_duration = 2.2
settings.sim_substeps = 100
settings.sim_dt = 5e-5
settings.initial_y = 0.08
settings.velocity_y = -0.05
experiment_name = "render_demo"
knife_height = 0.0305
device = "cuda"
requires_grad = False


sim = CuttingSim(settings, experiment_name=experiment_name,
                 adapter=device, requires_grad=requires_grad)
# disable gradient tracking for the entire simulation back-end to speed up the simulation
sim.disable_gradients()

sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.045, settings.initial_y, 0.0], device=device),
    linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

sim.cut()

# This simulation will save a USD file at outputs/render_demo.usd
# which can be viewed in usdview or Omniverse
sim.simulate(render=True)

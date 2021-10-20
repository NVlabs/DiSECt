# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This example demonstrates the basic cutting functionality of our simulator, and
will open an interactive 3D visualizer to show the simulation in real time.
"""

# fmt: off
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from cutting import load_settings, SlicingMotion, CuttingSim
# fmt: on

settings = load_settings("examples/config/ansys_prism.json")
settings.sim_duration = 5.0
settings.sim_dt = 1e-4  # 5e-5
settings.initial_y = 0.08
settings.velocity_y = -0.05
experiment_name = "cutting_prism"
knife_height = 0.0305
device = "cuda"
requires_grad = False


sim = CuttingSim(settings, experiment_name=experiment_name,
                 adapter=device, requires_grad=requires_grad)

slicing_kernel_width = 0.5
slicing_waypoints = 5
slicing_amplitudes = torch.tensor(np.ones(
    slicing_waypoints) * 0.1, device=device, dtype=torch.float32, requires_grad=requires_grad)
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
# sim.motion.plot(settings.sim_duration)
# import matplotlib.pyplot as plt
# plt.show()

sim.cut()

# sim.visualize_cut(cut_separation=0.01)

sim.visualize()

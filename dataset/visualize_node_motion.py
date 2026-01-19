import sys
import time

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5 import Qt
from threading import Thread
from tqdm import tqdm

import numpy as np
import pickle

import pyvista as pv
from pyvistaqt import BackgroundPlotter

import matplotlib.pyplot as plt
import seaborn
# seaborn.set()

app = Qt.QApplication(sys.argv)

plotter = BackgroundPlotter(title='Node Motion', auto_update=20.)          #, window_size=(1920, 1080))
plotter.show()

# log = pickle.load(open("results/ansys_sphere_apple_adam_gpu3_l1loss_ansys_node_motion_weight50000_1.pkl", "rb"))
# log = pickle.load(open("results/ansys_sphere_apple_adam_gpu3_l1loss_ansys_node_motion_weight50000_40.pkl", "rb"))
# log = pickle.load(open("results/ansys_sphere_apple_adam_gpu3_l1loss_ansys_node_motion_weight50000_264.pkl", "rb"))

iteration = 300
dataset = "ansys_sphere_apple"
dataset = "ansys_prism_potato"
# dataset = "ansys_cylinder_cucumber"
path = f"node_motion/{dataset}/iter_{iteration}"
os.makedirs(path, exist_ok=True)
log = pickle.load(
    open(
        f"results/ansys_prism_potato_adam_gpu1_l1loss_ansys_node_motion_prism_weight5000/ansys_prism_potato_adam_gpu1_l1loss_ansys_node_motion_prism_weight5000_{iteration}.pkl",
        "rb"))
# log = pickle.load(
#     open(
#         f"results/ansys_cylinder_cucumber_adam_gpu3_l1loss_ansys_node_motion_cylinder_weight5000/ansys_cylinder_cucumber_adam_gpu3_l1loss_ansys_node_motion_cylinder_weight5000_{iteration}.pkl",
#         "rb"))

timesteps = np.linspace(0.0, log["settings"]["sim_duration"], len(log["hist_knife_force"]))
plt.figure(figsize=(4,3))
print("Knife force MAE:", np.mean(np.abs(log["groundtruth"][:len(timesteps)] - log["hist_knife_force"][:len(timesteps)])))
plt.plot(timesteps, log["groundtruth"][:len(timesteps)], color="k", label="Groundtruth")
plt.plot(timesteps, log["hist_knife_force"][:len(timesteps)], '--', color="teal", label="Estimated")
plt.grid()
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Knife Force [N]")
# plt.title(f"Ansys Cylinder Cucumber - Iteration {iteration}")
plt.title(f"Ansys Prism Potato - Iteration {iteration}")
plt.tight_layout()
plt.savefig(f"{path}/node_motion_knife_{dataset}_i{iteration}.pdf")


mesh_filename = log["settings"]["generators"]["ansys"]["filename"]

initial_diff = log["node_motion"][1]["estimated"] - log["node_motion"][1]["groundtruth"]
initial_avg_offset = np.mean(initial_diff, axis=0)
# print("initial_diff", initial_diff)
# plotter.add_text("Average offset at t = %.3fs: " % log["node_motion"][1]["time"] + str(initial_avg_offset), font_size=10, position='lower_left')
# plotter.add_text("Average distance at t = %.3fs: " % log["node_motion"][1]["time"] + str(np.linalg.norm(initial_avg_offset)**2),
#                  font_size=10,
#                  position='lower_right')

estimated_points = pv.PolyData(log["node_motion"][1]["estimated"] - initial_avg_offset)
plotter.add_mesh(estimated_points, style='points', render_points_as_spheres=False, point_size=7, color="teal")

true_points = pv.PolyData(log["node_motion"][1]["groundtruth"])
plotter.add_mesh(true_points, style='points', render_points_as_spheres=False, point_size=7, color='black')

# screen_label = plotter.add_text("", font_size=10)

# ground plane
plotter.add_mesh(pv.Plane(direction=(0, 1, 0), i_size=0.11, j_size=0.11, i_resolution=11, j_resolution=11),
                 style='wireframe',
                 color=(0.3, 0.3, 0.3),
                 opacity=1,
                 edge_color=(0.3, 0.3, 0.3),
                 line_width=1,
                 show_edges=True,
                 render_lines_as_tubes=False)

# error lines
error_mesh = pv.PolyData(np.vstack((np.array(estimated_points.points), np.array(true_points.points))),
                         np.hstack([[2, i, len(estimated_points.points) + i] for i in range(len(estimated_points.points))]))
plotter.add_mesh(error_mesh,
                 style='wireframe',
                 color='red',
                 edge_color='red',
                 line_width=5,
                 show_edges=True,
                 render_lines_as_tubes=False)

# plotter.add_axes_at_origin(labels_off=True)
plotter.set_viewup([0., 1., 0.])
plotter.set_focus([0., 0., 0.])
plotter.set_position([0., .002, 0.001], True)
plotter.set_background("white")

# position, target, up vector
plotter.camera_position = ([0., .025, 0.25], [0.0, 0.01, 0.0], [0., 1., 0.])


# plotter.isometric_view()
for t in tqdm(sorted(log["node_motion"].keys())):
    motion = log["node_motion"][t]
    # screen_label.SetText(2, "Time: %02.2f" % motion["time"])

    print(f"Node motion MAE at t={t}:", np.mean(np.linalg.norm(motion["estimated"] - motion["groundtruth"], axis=1)))

    estimated_points.points = motion["estimated"] - initial_avg_offset
    true_points.points = motion["groundtruth"]

    error_mesh.points = np.vstack((estimated_points.points, true_points.points))

    rt = motion["time"]
    plotter.screenshot(f"{path}/node_motion_{dataset}_i{iteration}_{t:03d}.png", transparent_background=True, window_size=(1920, 1080))

# def simulate():
#     while plotter.app_window.isVisible():
#         for t in tqdm(sorted(log["node_motion"].keys())):
#             motion = log["node_motion"][t]
#             screen_label.SetText(2, "Time: %02.2f" % motion["time"])

#             estimated_points.points = motion["estimated"] - initial_avg_offset
#             true_points.points = motion["groundtruth"]
#             time.sleep(0.2)

# thread = Thread(target=simulate)
# thread.start()

# sys.exit(app.exec_())
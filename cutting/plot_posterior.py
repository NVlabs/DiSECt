# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helper functions for plotting."""

from scipy.interpolate import griddata
import matplotlib.collections as mcoll
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
sns.set(style="white")


def colorline(ax,
              x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0, zorder=0):
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def get_loss_functor(loss_fn):
    import torch
    if loss_fn == "l2":
        return torch.nn.MSELoss()
    elif loss_fn == "l1":
        return torch.nn.L1Loss()
    elif loss_fn == "smooth_l1":
        return torch.nn.SmoothL1Loss()
    elif loss_fn == "cosine":
        cos = torch.nn.CosineSimilarity(dim=0)
        return lambda x, y: 1.-cos(x, y)
    elif loss_fn == "logsumexp":
        return lambda x, y: torch.logsumexp((x-y)**2., dim=0)


def make_dataframe(log_format, iterations, is_groundtruth=False, progressbar=True, loss_fn: str = None, groundtruth=None):
    iterations = list(iterations)
    if groundtruth is not None and type(groundtruth) == str:
        groundtruth = pickle.load(open(groundtruth, "rb"))
        groundtruth = groundtruth["hist_knife_force"]
    log = pickle.load(open(log_format.format(iter=iterations[0]), "rb"))
    data = {key: [] for key in log["parameters"]}
    if loss_fn is not None and groundtruth is not None:
        import torch
        if type(loss_fn) == str:
            loss_fn = get_loss_functor(loss_fn)
        data["loss"] = []
        groundtruth = torch.tensor(groundtruth)
    if progressbar:
        import tqdm
        progress = tqdm.notebook.tqdm(iterations)
    else:
        progress = iterations
    for i in progress:
        log = pickle.load(open(log_format.format(iter=i), "rb"))
        for key, p in log["parameters"].items():
            if is_groundtruth:
                data[key].append(log["settings"][key])
            else:
                data[key].append(p)
        if loss_fn is not None and groundtruth is not None:
            min_len = min(len(log["hist_knife_force"]), len(groundtruth))
            data["loss"].append(loss_fn(torch.tensor(
                log["hist_knife_force"][:min_len]), groundtruth[:min_len]).item())
    data['iteration'] = iterations
    return pd.DataFrame(data)


def add_margin(ax, x=0.05, y=0.05):
    # This will, by default, add 5% to the x and y margins. You
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)


def plot_loss_landscape(ax, groundtruth_df, key_i: str, key_j: str, levels=10, cmap="RdBu_r", alpha=0.5):
    assert("loss" in groundtruth_df.columns)
    losses = np.array(groundtruth_df["loss"].values)
    xi = np.linspace(groundtruth_df[key_i].min(
    ), groundtruth_df[key_i].max(), len(losses))
    yi = np.linspace(groundtruth_df[key_j].min(
    ), groundtruth_df[key_j].max(), len(losses))
    zi = griddata((groundtruth_df[key_i], groundtruth_df[key_j]),
                  losses, (xi[None, :], yi[:, None]), method='cubic')
    CS = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=alpha)
    return CS


def plot_posterior(df, is_groundtruth=False, groundtruth_filename=None, parameters=None, groundtruth_df=None):
    frame = df[df.columns[~df.columns.isin(['iteration', 'loss'])]]
    g = sns.PairGrid(frame)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, bins=20, kde=True)
    if groundtruth_filename is not None:
        gt = pickle.load(open(groundtruth_filename, "rb"))
        gt = {key: gt["settings"][key]
              for key in frame.columns if key != 'iteration'}
        for i, key_i in enumerate(frame.columns):
            for j, key_j in enumerate(frame.columns):
                if i == j:
                    g.axes[i, j].axvline(gt[key_i], color="r")
                    if parameters is not None and key_i in parameters:
                        param = parameters[key_i]
                        g.axes[i, i].set_xlim((param.low, param.high))
                elif j < i:
                    g.axes[i, j].scatter(
                        [gt[key_j]], [gt[key_i]], marker='*', color="r")
                    if parameters is not None and key_i in parameters:
                        param = parameters[key_i]
                        g.axes[i, j].set_ylim((param.low, param.high))
                add_margin(g.axes[i, j], x=0.02, y=0.02)
    for i, key_i in enumerate(frame.columns):
        for j, key_j in enumerate(frame.columns):
            if i < j:
                g.axes[i, j].grid()
                if (groundtruth_df is not None
                        and parameters is not None
                        and key_i in parameters
                        and key_j in parameters
                        and "loss" in groundtruth_df.columns):
                    plot_loss_landscape(
                        g.axes[i, j], groundtruth_df, key_j, key_i)
                    if groundtruth_filename is not None:
                        g.axes[i, j].scatter(
                            [gt[key_j]], [gt[key_i]], marker='*', color="black", s=100)
                if parameters is not None and key_i in parameters:
                    param = parameters[key_i]
                    g.axes[i, j].set_ylim((param.low, param.high))
                else:
                    g.axes[i, j].set_ylim(
                        (frame[key_i].min(), frame[key_i].max()))
                if not is_groundtruth:
                    g.axes[i, j].scatter(
                        frame[key_j], frame[key_i], marker='.', color="k", s=20)
                    colorline(g.axes[i, j], frame[key_j],
                              frame[key_i], linewidth=1.2, alpha=0.8)
                    if groundtruth_filename is not None:
                        g.axes[i, j].scatter(
                            [gt[key_j]], [gt[key_i]], marker='*', color="r", zorder=2)
                else:
                    g.axes[i, j].scatter(
                        frame[key_j], frame[key_i], marker='.', color="k", alpha=0.5, s=0.5)
    return g


def plot_loss_eval(groundtruth_df, log_format, gt_file_format, gt_id=20, log_range=range(1, 201), dim_i=0, dim_j=1):
    from IPython.display import display
    import tqdm
    import torch

    log = pickle.load(open(log_format.format(iter=1), "rb"))
    loss_fn = log["training"]["loss_fn"]
    loss_functor = get_loss_functor(loss_fn)

    param_name_i = list(log["parameters"].keys())[dim_i]
    param_name_j = list(log["parameters"].keys())[dim_j]

    gt_params = []
    best_dist = 1e6
    best_i = 0
    gt_params_i = groundtruth_df[param_name_i].values
    gt_params_j = groundtruth_df[param_name_j].values
    for i in range(len(groundtruth_df.index)):
        p = np.array([gt_params_i[i], gt_params_j[i]])
        gt_params.append(p)

    gt_params = np.array(gt_params)
    gt_params_target = gt_params[gt_id]

    print(f'Iteration {list(log_range)[-1]} with {loss_fn.upper()} loss:')
    table = []
    log = pickle.load(open(log_format.format(iter=list(log_range)[-1]), "rb"))
    for key, val_log in log["parameters"].items():
        gt_val = groundtruth_df[key].values[gt_id]
        table.append([key, val_log, gt_val])
    df = pd.DataFrame(table, columns=["Parameter", "Log value", "Groundtruth"])
    display(df)

    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.grid()
    contour = plot_loss_landscape(
        plt.gca(), groundtruth_df, param_name_i, param_name_j)
    plt.colorbar(contour)
    plt.scatter(gt_params[:, 0], gt_params[:, 1], c='k', s=1, alpha=0.7)

    plt.scatter([gt_params_target[0]], [gt_params_target[1]],
                s=200, marker='*', c="black")
    plt.scatter([gt_params_target[0]], [gt_params_target[1]],
                s=100, marker='*', c="yellow", label="Groundtruth")

    log_params = []
    log_losses = []
    log_losses_real = []
    for i in tqdm.notebook.tqdm(log_range):
        log = pickle.load(open(log_format.format(iter=i), "rb"))
        p = np.array([log['parameters'][param_name_i],
                      log['parameters'][param_name_j]])
        log_params.append(p)
        log_losses.append(log['loss'])
        lg_force = np.array(log['hist_knife_force'])
        gt_force = log["groundtruth"]
        log_losses_real.append(loss_functor(
            torch.tensor(lg_force), torch.tensor(gt_force)))
    log_params = np.array(log_params)

    target_params = log_params[-1, :]
    distances = []
    for i in range(len(groundtruth_df.index)):
        p = gt_params[i, :]
        dist = np.linalg.norm(p - target_params)
        if dist < best_dist:
            best_dist = dist
            best_i = i
        distances.append(dist)
    distances = np.array(distances)

#     plt.plot(log_params[:,dim_i],
#              log_params[:,dim_j],
#              linewidth=4, color='white')

    colorline(plt.gca(), log_params[:, 0], log_params[:, 1],
              z=None, cmap=plt.get_cmap('jet'), linewidth=3)
    plt.plot(log_params[:, 0], log_params[:, 1], alpha=0)
#     plt.plot(log_params[:,dim_i],
#              log_params[:,dim_j],
#              '.-', markersize=3, linewidth=0.5)
    plt.scatter([log_params[-1, 0]], [log_params[-1, 1]],
                c='r', label="Final estimate")
    plt.legend(loc="upper right")

    plt.xlabel(param_name_i)
    plt.ylabel(param_name_j)
    plt.title(loss_fn.upper() + " Loss Landscape")

    plt.subplot(222)
    plt.grid()
    plt.plot(log_losses_real, label="Raw loss", color="k", linewidth=2)
    plt.plot(log_losses, '--', label="Reported loss", color="C1", linewidth=3)
    plt.title(loss_fn.upper() + " Loss Evolution")
    plt.legend()

    plt.subplot(223)
    plt.title("Distance to Last Parameter Estimate")
    plt.grid()
    contour = plt.tricontourf(gt_params[:, 0], gt_params[:, 1], distances,
                              alpha=0.5, levels=10, cmap="RdBu_r")
    plt.colorbar(contour)

    plt.scatter(gt_params[:, 0], gt_params[:, 1], c='k', s=1, alpha=0.7)
    plt.scatter([gt_params_target[0]], [gt_params_target[1]],
                s=200, marker='*', c="black")
    plt.scatter([gt_params_target[0]], [gt_params_target[1]],
                s=100, marker='*', c="yellow", label="Groundtruth")
    best_p = gt_params[best_i, :]
    plt.scatter([best_p[0]], [best_p[1]], color="r", s=40,
                label=f"Closest parameters ({best_i})")
    plt.scatter([log_params[-1, 0]], [log_params[-1, 1]],
                color="pink", s=40, label="Final estimate")
    colorline(plt.gca(), log_params[:, 0], log_params[:, 1],
              z=None, cmap=plt.get_cmap('jet'), linewidth=3)

    plt.xlabel(param_name_i)
    plt.ylabel(param_name_j)

    plt.legend(loc="upper right")

    plt.subplot(224)
    plt.title(f'Knife Force Profile')

    log = pickle.load(open(log_format.format(iter=list(log_range)[0]), "rb"))
    lg_force = np.array(log['hist_knife_force'])
    plt.plot(lg_force, "--", color="gray",
             label=f"Initial estimate (Iteration {list(log_range)[0]})")

    log = pickle.load(open(log_format.format(iter=list(log_range)[-1]), "rb"))
    lg_force = np.array(log['hist_knife_force'])
    plt.plot(
        lg_force, label=f"Final estimate (Iteration {list(log_range)[-1]})")

    glog = pickle.load(open(gt_file_format.format(iter=best_i), "rb"))
    glg_force = np.array(glog['hist_knife_force'])
    plt.plot(glg_force, label="Closest parameters")

    plt.plot(log['groundtruth'], label="Groundtruth")
    plt.legend()

    plt.tight_layout()

    return plt

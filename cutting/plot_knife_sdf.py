# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Plotting example to develop SDF for the knife shape."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def ndot(ax, ay, bx, by):
    return ax * bx - ay * by


def length(x, y):
    return np.sqrt(x * x + y * y)


def length3(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


# h is the height (along y)
# def sdf_knife(point, size, spine_dim=2., spine_height=40., edge_dim=0.08, edge_height=0.04, radius=0.05):
def sdf_knife(point, size, spine_dim=20., spine_height=70., edge_dim=10, tip_height=4, radius=0.0):
    px = np.abs(point[0])
    py = point[1]

    v0x = spine_dim * 0.5
    v0y = spine_height * 0.5
    v1x = edge_dim * 0.5
    v1y = -spine_height * 0.5
    v2x = 0.
    v2y = v1y - tip_height
    v3x = 0.
    v3y = v0y

    d = (px - v1x)**2. + (py - v1y)**2.

    # i = 0, j = 3 (simplified because v0 and v3 have same y coordinate)
    ex = v3x - v0x
    wx = px - v0x
    wy = py - v0y
    delta = np.clip(wx / ex, 0., 1.)
    bx = wx - ex * delta
    by = wy
    d = min(d, bx*bx + by*by)
    s1 = np.sign(0.0 - wy * ex)

    # i = 1, j = 0
    ex = v0x - v1x
    ey = v0y - v1y
    wx = px - v1x
    wy = py - v1y
    delta = np.clip((wx*ex + wy*ey) / (ex*ex + ey*ey), 0., 1.)
    bx = wx - ex * delta
    by = wy - ey * delta
    d = min(d, bx*bx + by*by)
    s2 = np.sign(wx * ey - wy * ex)

    # i = 2, j = 1
    ex = v1x - v2x
    ey = v1y - v2y
    wx = px - v2x
    wy = py - v2y
    delta = np.clip((wx*ex + wy*ey) / (ex*ex + ey*ey), 0., 1.)
    bx = wx - ex * delta
    by = wy - ey * delta
    d = min(d, bx*bx + by*by)
    s3 = np.sign(wx * ey - wy * ex)

    # i = 3, j = 2 (only for intersection test, v2 and v3 have same x coordinate)
    ey = v2y - v3y
    wx = px - v3x
    s4 = np.sign(wx * ey)

    s = s1 + s2 + s3 + s4
    s = -np.sign(abs(s) - 4) * 2 - 1
    return s * np.sqrt(d)


def gradient(func, point, eps=1e-6):
    dx0 = func(np.array([point[0] - eps, point[1], point[2]]))
    dx1 = func(np.array([point[0] + eps, point[1], point[2]]))
    dy0 = func(np.array([point[0], point[1] - eps, point[2]]))
    dy1 = func(np.array([point[0], point[1] + eps, point[2]]))
    dz0 = func(np.array([point[0], point[1], point[2] - eps]))
    dz1 = func(np.array([point[0], point[1], point[2] + eps]))
    eps2 = 2.0 * eps
    return np.array([(dx1 - dx0) / eps2, (dy1 - dy0) / eps2, (dz1 - dz0) / eps2])


def gradient_1d(func, x, eps=1e-6):
    dx0 = func(x - eps)
    dx1 = func(x + eps)
    eps2 = 2.0 * eps
    return (dx1 - dx0) / eps2


def frank_wolfe(sdf, p1, p2, num_iter=500):
    def func(u):
        return sdf((1. - u) * p1 + u * p2)

    u = 0.5
    for i in range(num_iter):
        print(f"Iter {i}\t{u}")
        grad = gradient_1d(func, u)
        if grad < 0.:
            s = 1.
        else:
            s = 0.
        s = max(np.sign(0.0 - grad), 0.0)
        gamma = 2 / (2 + i)
        u += gamma * (s - u)
    print("Shortest distance:", func(u))
    return u


num_samples = 300
limit = 60.
size = [1.8, 2.8, 1.]
a = np.zeros((num_samples, num_samples))
pbar = tqdm(total=num_samples**2)
for i, y in enumerate(np.linspace(limit, -limit, num_samples)):
    for j, x in enumerate(np.linspace(-limit, limit, num_samples)):
        a[i, j] = sdf_knife([x, y, 0.], size)
        pbar.update()
plt.imshow(a, cmap='RdYlBu', interpolation='nearest', extent=(-limit, limit, -limit, limit))
ax = plt.gca()
ax.contour(a, levels=[0], colors='k', linestyles='-', extent=(-limit, limit, limit, -limit))
ax.contour(a, levels=[5], colors='g', linestyles='--', extent=(-limit, limit, limit, -limit))
plt.colorbar()

p1 = np.array([-17, -48, 0.0])
p2 = np.array([25, -50, 0.0])
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '.-', c='m')
closest_u = frank_wolfe(lambda p: sdf_knife(p, size), p1, p2)
closest = (1. - closest_u) * p1 + closest_u * p2
ax.scatter([closest[0]], [closest[1]], c='b')

plt.figure()
us = np.linspace(0., 1., 100)
plt.grid()
plt.plot(us, [sdf_knife((1. - u) * p1 + u * p2, size) for u in us])
plt.scatter([closest_u], [sdf_knife(closest, size)], c='r', s=20)
plt.gca().set_xlabel('u')

plt.show()
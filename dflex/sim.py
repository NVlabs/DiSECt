# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This module contains time-integration objects for simulating
models + state forward in time.
"""

import math
import torch
import numpy as np

import dflex.util
import dflex.adjoint as df
import dflex.config

from dflex.model import *
import time


# externally compiled kernels module (C++/CUDA code with PyBind entry points)
kernels = None


@df.func
def test(c: float):

    x = 1.0
    y = float(2)
    z = int(3.0)

    print(y)
    print(z)

    if (c < 3.0):
        x = 2.0

    return x * 6.0


def kernel_init():
    global kernels
    kernels = df.compile()


@df.kernel
def integrate_particles(x: df.tensor(df.float3),
                        v: df.tensor(df.float3),
                        f: df.tensor(df.float3),
                        w: df.tensor(float),
                        gravity: df.tensor(df.float3),
                        dt: float,
                        x_new: df.tensor(df.float3),
                        v_new: df.tensor(df.float3)):

    tid = df.tid()

    x0 = df.load(x, tid)
    v0 = df.load(v, tid)
    f0 = df.load(f, tid)
    inv_mass = df.load(w, tid)

    g = df.load(gravity, 0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g * df.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    df.store(x_new, tid, x1)
    df.store(v_new, tid, v1)


# semi-implicit Euler integration
@df.kernel
def integrate_rigids(rigid_x: df.tensor(df.float3),
                     rigid_r: df.tensor(df.quat),
                     rigid_v: df.tensor(df.float3),
                     rigid_w: df.tensor(df.float3),
                     rigid_f: df.tensor(df.float3),
                     rigid_t: df.tensor(df.float3),
                     inv_m: df.tensor(float),
                     inv_I: df.tensor(df.mat33),
                     gravity: df.tensor(df.float3),
                     dt: float,
                     rigid_x_new: df.tensor(df.float3),
                     rigid_r_new: df.tensor(df.quat),
                     rigid_v_new: df.tensor(df.float3),
                     rigid_w_new: df.tensor(df.float3)):

    tid = df.tid()

    # positions
    x0 = df.load(rigid_x, tid)
    r0 = df.load(rigid_r, tid)

    # velocities
    v0 = df.load(rigid_v, tid)
    w0 = df.load(rigid_w, tid)         # angular velocity

    # forces
    f0 = df.load(rigid_f, tid)
    t0 = df.load(rigid_t, tid)

    # masses
    inv_mass = df.load(inv_m, tid)     # 1 / mass
    inv_inertia = df.load(inv_I, tid)  # inverse of 3x3 inertia matrix

    g = df.load(gravity, 0)

    # linear part
    v1 = v0 + (f0 * inv_mass + g * df.nonzero(inv_mass)) * \
        dt        # linear integral (linear position/velocity)
    x1 = x0 + v1 * dt

    # angular part

    # so reverse multiplication by r0 takes you from global coordinates into local coordinates
    # because it's covector and thus gets pulled back rather than pushed forward
    # angular integral (angular velocity and rotation), rotate into object reference frame
    wb = df.rotate_inv(r0, w0)
    # also rotate torques into local coordinates
    tb = df.rotate_inv(r0, t0)

    # I^{-1} torque = angular acceleration and inv_inertia is always going to be in the object frame.
    # So we need to rotate into that frame, and then back into global.
    # I^-1 * torque * dt., then go back into global coordinates
    w1 = df.rotate(r0, wb + inv_inertia * tb * dt)
    r1 = df.normalize(r0 + df.quat(w1, 0.0) * r0 * 0.5 *
                      dt)         # rotate around w1 by dt

    df.store(rigid_x_new, tid, x1)
    df.store(rigid_r_new, tid, r1)
    df.store(rigid_v_new, tid, v1)
    df.store(rigid_w_new, tid, w1)


@df.kernel
def eval_springs(x: df.tensor(df.float3),
                 v: df.tensor(df.float3),
                 spring_indices: df.tensor(int),
                 spring_rest_lengths: df.tensor(float),
                 spring_stiffness: df.tensor(float),
                 spring_damping: df.tensor(float),
                 f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(spring_indices, tid * 2 + 0)
    j = df.load(spring_indices, tid * 2 + 1)

    ke = df.load(spring_stiffness, tid)
    kd = df.load(spring_damping, tid)
    rest = df.load(spring_rest_lengths, tid)

    xi = df.load(x, i)
    xj = df.load(x, j)

    vi = df.load(v, i)
    vj = df.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    df.atomic_sub(f, i, fs)
    df.atomic_add(f, j, fs)


@df.kernel
def eval_triangles(x: df.tensor(df.float3),
                   v: df.tensor(df.float3),
                   indices: df.tensor(int),
                   pose: df.tensor(df.mat22),
                   activation: df.tensor(float),
                   k_mu: float,
                   k_lambda: float,
                   k_damp: float,
                   k_drag: float,
                   k_lift: float,
                   f: df.tensor(df.float3)):
    tid = df.tid()

    i = df.load(indices, tid * 3 + 0)
    j = df.load(indices, tid * 3 + 1)
    k = df.load(indices, tid * 3 + 2)

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    vp = df.load(v, i)       # vel zero
    vq = df.load(v, j)       # vel one
    vr = df.load(v, k)       # vel two

    qp = q - p     # barycentric coordinates (centered at p)
    rp = r - p

    Dm = df.load(pose, tid)

    inv_rest_area = df.determinant(Dm) * 2.0     # 1 / det(A) = det(A^-1)
    rest_area = 1.0 / inv_rest_area

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    f1 = qp * Dm[0, 0] + rp * Dm[1, 0]
    f2 = qp * Dm[0, 1] + rp * Dm[1, 1]

    # -----------------------------
    # St. Venant-Kirchoff

    # # Green strain, F'*F-I
    # e00 = dot(f1, f1) - 1.0
    # e10 = dot(f2, f1)
    # e01 = dot(f1, f2)
    # e11 = dot(f2, f2) - 1.0

    # E = df.mat22(e00, e01,
    #              e10, e11)

    # # local forces (deviatoric part)
    # T = df.mul(E, df.transpose(Dm))

    # # spatial forces, F*T
    # fq = (f1*T[0,0] + f2*T[1,0])*k_mu*2.0
    # fr = (f1*T[0,1] + f2*T[1,1])*k_mu*2.0
    # alpha = 1.0

    # -----------------------------
    # Baraff & Witkin, note this model is not isotropic

    # c1 = length(f1) - 1.0
    # c2 = length(f2) - 1.0
    # f1 = normalize(f1)*c1*k1
    # f2 = normalize(f2)*c2*k1

    # fq = f1*Dm[0,0] + f2*Dm[0,1]
    # fr = f1*Dm[1,0] + f2*Dm[1,1]

    # -----------------------------
    # Neo-Hookean (with rest stability)

    # force = mu*F*Dm'
    fq = (f1 * Dm[0, 0] + f2 * Dm[0, 1]) * k_mu
    fr = (f1 * Dm[1, 0] + f2 * Dm[1, 1]) * k_mu
    alpha = 1.0 + k_mu / k_lambda

    # -----------------------------
    # Area Preservation

    n = df.cross(qp, rp)
    area = df.length(n) * 0.5

    # actuation
    act = df.load(activation, tid)

    # J-alpha
    c = area * inv_rest_area - alpha + act

    # dJdx
    n = df.normalize(n)
    dcdq = df.cross(rp, n) * inv_rest_area * 0.5
    dcdr = df.cross(n, qp) * inv_rest_area * 0.5

    f_area = k_lambda * c

    # -----------------------------
    # Area Damping

    dcdt = dot(dcdq, vq) + dot(dcdr, vr) - dot(dcdq + dcdr, vp)
    f_damp = k_damp * dcdt

    fq = fq + dcdq * (f_area + f_damp)
    fr = fr + dcdr * (f_area + f_damp)
    fp = fq + fr

    # -----------------------------
    # Lift + Drag

    vmid = (vp + vr + vq) * 0.3333
    vdir = df.normalize(vmid)

    f_drag = vmid * (k_drag * area * df.abs(df.dot(n, vmid)))
    f_lift = n * (k_lift * area * (1.57079 -
                  df.acos(df.dot(n, vdir)))) * dot(vmid, vmid)

    # note reversed sign due to atomic_add below.. need to write the unary op -
    fp = fp - f_drag - f_lift
    fq = fq + f_drag + f_lift
    fr = fr + f_drag + f_lift

    # apply forces
    df.atomic_add(f, i, fp)
    df.atomic_sub(f, j, fq)
    df.atomic_sub(f, k, fr)


@df.func
def triangle_closest_point_barycentric(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = df.dot(ab, ap)
    d2 = df.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return float3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = df.dot(ab, bp)
    d4 = df.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return float3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return float3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return float3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return float3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return float3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return float3(1.0 - v - w, v, w)


# @df.func
# def triangle_closest_point(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
#     ab = b - a
#     ac = c - a
#     ap = p - a

#     d1 = df.dot(ab, ap)
#     d2 = df.dot(ac, ap)

#     if (d1 <= 0.0 and d2 <= 0.0):
#         return a

#     bp = p - b
#     d3 = df.dot(ab, bp)
#     d4 = df.dot(ac, bp)

#     if (d3 >= 0.0 and d4 <= d3):
#         return b

#     vc = d1 * d4 - d3 * d2
#     v = d1 / (d1 - d3)
#     if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
#         return a + ab * v

#     cp = p - c
#     d5 = dot(ab, cp)
#     d6 = dot(ac, cp)

#     if (d6 >= 0.0 and d5 <= d6):
#         return c

#     vb = d5 * d2 - d1 * d6
#     w = d2 / (d2 - d6)
#     if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
#         return a + ac * w

#     va = d3 * d6 - d5 * d4
#     w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
#     if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
#         return b + (c - b) * w

#     denom = 1.0 / (va + vb + vc)
#     v = vb * denom
#     w = vc * denom

#     return a + ab * v + ac * w


@df.kernel
def eval_triangles_contact(
        # idx : df.tensor(int), # list of indices for colliding particles
        num_particles: int,                # size of particles
        x: df.tensor(df.float3),
        v: df.tensor(df.float3),
        indices: df.tensor(int),
        pose: df.tensor(df.mat22),
        activation: df.tensor(float),
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        k_drag: float,
        k_lift: float,
        f: df.tensor(df.float3)):

    tid = df.tid()
    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # index = df.load(idx, tid)
    pos = df.load(x, particle_no)      # at the moment, just one particle
    # vel0 = df.load(v, 0)

    i = df.load(indices, face_no * 3 + 0)
    j = df.load(indices, face_no * 3 + 1)
    k = df.load(indices, face_no * 3 + 2)

    if (i == particle_no or j == particle_no or k == particle_no):
        return

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    # vp = df.load(v, i) # vel zero
    # vq = df.load(v, j) # vel one
    # vr = df.load(v, k)  # vel two

    # qp = q-p # barycentric coordinates (centered at p)
    # rp = r-p

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = df.dot(diff, diff)
    n = df.normalize(diff)
    c = df.min(dist - 0.01, 0.0)       # 0 unless within 0.01 of surface
    #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    fn = n * c * 1e5

    df.atomic_sub(f, particle_no, fn)

    # # apply forces (could do - f / 3 here)
    df.atomic_add(f, i, fn * bary[0])
    df.atomic_add(f, j, fn * bary[1])
    df.atomic_add(f, k, fn * bary[2])


@df.kernel
def eval_triangles_rigid_contacts(
        # number of particles (size of contact_point)
        num_particles: int,
        x: df.tensor(df.float3),                     # position of particles
        v: df.tensor(df.float3),
        indices: df.tensor(int),                     # triangle indices
        rigid_x: df.tensor(df.float3),               # rigid body positions
        rigid_r: df.tensor(df.quat),
        rigid_v: df.tensor(df.float3),
        rigid_w: df.tensor(df.float3),
        contact_body: df.tensor(int),
        # position of contact points relative to body
        contact_point: df.tensor(df.float3),
        contact_dist: df.tensor(float),
        contact_mat: df.tensor(int),
        materials: df.tensor(float),
        #   rigid_f : df.tensor(df.float3),
    #   rigid_t : df.tensor(df.float3),
        tri_f: df.tensor(df.float3)):

    tid = df.tid()

    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # -----------------------
    # load rigid body point
    c_body = df.load(contact_body, particle_no)
    c_point = df.load(contact_point, particle_no)
    c_dist = df.load(contact_dist, particle_no)
    c_mat = df.load(contact_mat, particle_no)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = df.load(rigid_x, c_body)      # position of colliding body
    r0 = df.load(rigid_r, c_body)      # orientation of colliding body

    v0 = df.load(rigid_v, c_body)
    w0 = df.load(rigid_w, c_body)

    # transform point to world space
    pos = x0 + df.rotate(r0, c_point)
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = pos - x0                       # basically just c_point in the new coordinates
    rhat = df.normalize(r)
    # add on 'thickness' of shape, e.g.: radius of sphere/capsule
    pos = pos + rhat * c_dist

    # contact point velocity
    # this is rigid velocity cross offset, so it's the velocity of the contact point.
    dpdt = v0 + df.cross(w0, r)

    # -----------------------
    # load triangle
    i = df.load(indices, face_no * 3 + 0)
    j = df.load(indices, face_no * 3 + 1)
    k = df.load(indices, face_no * 3 + 2)

    p = df.load(x, i)        # point zero
    q = df.load(x, j)        # point one
    r = df.load(x, k)        # point two

    vp = df.load(v, i)       # vel zero
    vq = df.load(v, j)       # vel one
    vr = df.load(v, k)       # vel two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest               # vector from tri to point
    dist = df.dot(diff, diff)          # squared distance
    n = df.normalize(diff)             # points into the object
    c = df.min(dist - 0.05, 0.0)       # 0 unless within 0.05 of surface
    #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # df.atomic_sub(tri_f, particle_no, fn)

    # normal force (restitution coefficient * how far inside for ground) (negative)
    fn = c * ke

    # bad approximation for centroid velocity
    vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]
    vrel = vtri - dpdt

    # velocity component of rigid in negative normal direction
    vn = dot(n, vrel)
    vt = vrel - n * vn       # velocity component not in normal direction

    # contact damping
    # again, negative, into the ground
    fd = 0.0 - df.max(vn, 0.0) * kd * df.step(c)

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = 0.0 - lower      # workaround because no unary ops yet

    nx = cross(n, float3(0.0, 0.0, 1.0))         # basis vectors for tangent
    nz = cross(n, float3(1.0, 0.0, 0.0))

    vx = df.clamp(dot(nx * kf, vt), lower, upper)
    vz = df.clamp(dot(nz * kf, vt), lower, upper)

    # df.float3(vx, 0.0, vz)*df.step(c)
    ft = (nx * vx + nz * vz) * (0.0 - df.step(c))

    # # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft

    df.atomic_add(tri_f, i, f_total * bary[0])
    df.atomic_add(tri_f, j, f_total * bary[1])
    df.atomic_add(tri_f, k, f_total * bary[2])


@df.kernel
def eval_bending(
        x: df.tensor(df.float3), v: df.tensor(df.float3), indices: df.tensor(int), rest: df.tensor(float), ke: float, kd: float, f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(indices, tid * 4 + 0)
    j = df.load(indices, tid * 4 + 1)
    k = df.load(indices, tid * 4 + 2)
    l = df.load(indices, tid * 4 + 3)

    rest_angle = df.load(rest, tid)

    x1 = df.load(x, i)
    x2 = df.load(x, j)
    x3 = df.load(x, k)
    x4 = df.load(x, l)

    v1 = df.load(v, i)
    v2 = df.load(v, j)
    v3 = df.load(v, k)
    v4 = df.load(v, l)

    n1 = df.cross(x3 - x1, x4 - x1)    # normal to face 1
    n2 = df.cross(x4 - x2, x3 - x2)    # normal to face 2

    n1_length = df.length(n1)
    n2_length = df.length(n2)

    rcp_n1 = 1.0 / n1_length
    rcp_n2 = 1.0 / n2_length

    cos_theta = df.dot(n1, n2) * rcp_n1 * rcp_n2

    n1 = n1 * rcp_n1 * rcp_n1
    n2 = n2 * rcp_n2 * rcp_n2

    e = x4 - x3
    e_hat = df.normalize(e)
    e_length = df.length(e)

    s = df.sign(df.dot(df.cross(n2, n1), e_hat))
    angle = df.acos(cos_theta) * s

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * df.dot(x1 - x4, e_hat) + n2 * df.dot(x2 - x4, e_hat)
    d4 = n1 * df.dot(x3 - x1, e_hat) + n2 * df.dot(x3 - x2, e_hat)

    # elastic
    f_elastic = ke * (angle - rest_angle)

    # damping
    f_damp = kd * (df.dot(d1, v1) + df.dot(d2, v2) +
                   df.dot(d3, v3) + df.dot(d4, v4))

    # total force, proportional to edge length
    f_total = 0.0 - e_length * (f_elastic + f_damp)

    df.atomic_add(f, i, d1 * f_total)
    df.atomic_add(f, j, d2 * f_total)
    df.atomic_add(f, k, d3 * f_total)
    df.atomic_add(f, l, d4 * f_total)


@df.kernel
def eval_tetrahedra(x: df.tensor(df.float3),
                    v: df.tensor(df.float3),
                    indices: df.tensor(int),
                    pose: df.tensor(df.mat33),
                    activation: df.tensor(float),
                    tet_mu: df.tensor(float),
                    tet_lambda: df.tensor(float),
                    tet_damping: df.tensor(float),
                    f: df.tensor(df.float3)):

    tid = df.tid()

    i = df.load(indices, tid * 4 + 0)
    j = df.load(indices, tid * 4 + 1)
    k = df.load(indices, tid * 4 + 2)
    l = df.load(indices, tid * 4 + 3)

    act = df.load(activation, tid)

    k_mu = df.load(tet_mu, tid)
    k_lambda = df.load(tet_lambda, tid)
    k_damp = df.load(tet_damping, tid)

    x0 = df.load(x, i)
    x1 = df.load(x, j)
    x2 = df.load(x, k)
    x3 = df.load(x, l)

    v0 = df.load(v, i)
    v1 = df.load(v, j)
    v2 = df.load(v, k)
    v3 = df.load(v, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = df.mat33(x10, x20, x30)
    Dm = df.load(pose, tid)

    inv_rest_volume = df.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = df.mat33(v10, v20, v30) * Dm

    # -----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])
    Ic = F[0, 0] * F[0, 0] + F[1, 1] * F[1, 1] + F[2, 2] * F[2, 2]

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * df.transpose(Dm)

    f1 = df.float3(H[0, 0], H[1, 0], H[2, 0])
    f2 = df.float3(H[0, 1], H[1, 1], H[2, 1])
    f3 = df.float3(H[0, 2], H[1, 2], H[2, 2])

    # # hydrostatic part
    J = df.determinant(F)

    # print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = df.cross(x20, x30) * s
    dJdx2 = df.cross(x30, x10) * s
    dJdx3 = df.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (df.dot(dJdx1, v1) + df.dot(dJdx2, v2) +
              df.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = (f1 + f2 + f3) * (0.0 - 1.0)

    # apply forces
    df.atomic_sub(f, i, f0)
    df.atomic_sub(f, j, f1)
    df.atomic_sub(f, k, f2)
    df.atomic_sub(f, l, f3)


@df.kernel
def eval_contacts(x: df.tensor(df.float3),
                  v: df.tensor(df.float3),
                  ke: float,
                  kd: float,
                  kf: float,
                  mu: float,
                  closeness: float,
                  mask: df.tensor(float),
                  f: df.tensor(df.float3)):

    # this just handles contact of particles with the ground plane, nothing else.
    tid = df.tid()

    x0 = df.load(x, tid)
    v0 = df.load(v, tid)

    weight = df.load(mask, tid)

    # why is the normal always y? Ground is always (0, 1, 0) normal
    n = float3(0.0, 1.0, 0.0)

    # 0 unless within 0.01 of surface
    c = df.min(dot(n, x0) - closeness, 0.0)
    #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)

    vn = dot(n, v0)
    vt = v0 - n * vn

    fn = n * c * ke

    # contact damping
    fd = n * df.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    ftotal = fn + (fd + ft) * df.step(c)

    ftotal = ftotal * weight

    df.atomic_sub(f, tid, ftotal)


@df.kernel
def eval_hard_particle_contacts(edge_indices: df.tensor(int),
                                edge_coords: df.tensor(float),
                                x: df.tensor(df.float3),
                                v: df.tensor(df.float3),
                                ke: float,
                                kd: float,
                                kf: float,
                                mu: float,
                                closeness: float,
                                f: df.tensor(df.float3)):

    # this just handles contact of particles with the ground plane, nothing else.
    tid = df.tid()

    i = df.load(edge_indices, tid * 2 + 0)
    j = df.load(edge_indices, tid * 2 + 1)
    t = df.load(edge_coords, tid)

    xi = df.load(x, i)
    xj = df.load(x, j)
    vi = df.load(v, i)
    vj = df.load(v, j)

    x0 = xi * (1.0 - t) + xj * t
    v0 = vi * (1.0 - t) + vj * t

    # why is the normal always y? Ground is always (0, 1, 0) normal
    n = float3(0.0, 1.0, 0.0)

    # 0 unless within 0.01 of surface
    c = df.min(dot(n, x0) - closeness, 0.0)
    #c = df.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)

    vn = dot(n, v0)
    vt = v0 - n * vn

    fn = n * c * ke

    # contact damping
    fd = n * df.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    ftotal = fn + (fd + ft) * df.step(c)

    fi = ftotal * (1.0 - t)
    fj = ftotal * t

    df.atomic_sub(f, i, fi)
    df.atomic_sub(f, j, fj)


@df.func
def sphere_sdf(center: df.float3, radius: float, p: df.float3):

    return df.length(p - center) - radius


@df.func
def sphere_sdf_grad(center: df.float3, radius: float, p: df.float3):

    return df.normalize(p - center)


@df.func
def box_sdf(upper: df.float3, p: df.float3):

    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    e = df.float3(df.max(qx, 0.0), df.max(qy, 0.0), df.max(qz, 0.0))

    return df.length(e) + df.min(df.max(qx, df.max(qy, qz)), 0.0)


@df.func
def box_sdf_grad(upper: df.float3, p: df.float3):

    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    # exterior case
    if (qx > 0.0 or qy > 0.0 or qz > 0.0):

        x = df.clamp(p[0], 0.0 - upper[0], upper[0])
        y = df.clamp(p[1], 0.0 - upper[1], upper[1])
        z = df.clamp(p[2], 0.0 - upper[2], upper[2])

        return df.normalize(p - df.float3(x, y, z))

    sx = df.sign(p[0])
    sy = df.sign(p[1])
    sz = df.sign(p[2])

    # x projection
    if (qx > qy and qx > qz):
        return df.float3(sx, 0.0, 0.0)

    # y projection
    if (qy > qx and qy > qz):
        return df.float3(0.0, sy, 0.0)

    # z projection
    if (qz > qx and qz > qy):
        return df.float3(0.0, 0.0, sz)


@df.func
def capsule_sdf(radius: float, half_width: float, p: df.float3):

    if (p[0] > half_width):
        return length(df.float3(p[0] - half_width, p[1], p[2])) - radius

    if (p[0] < 0.0 - half_width):
        return length(df.float3(p[0] + half_width, p[1], p[2])) - radius

    return df.length(df.float3(0.0, p[1], p[2])) - radius


@df.func
def capsule_sdf_grad(radius: float, half_width: float, p: df.float3):

    if (p[0] > half_width):
        return normalize(df.float3(p[0] - half_width, p[1], p[2]))

    if (p[0] < 0.0 - half_width):
        return normalize(df.float3(p[0] + half_width, p[1], p[2]))

    return normalize(df.float3(0.0, p[1], p[2]))


@df.func
def knife_sdf(point: df.float3, spine_dim: float, spine_height: float, edge_dim: float, tip_height: float, depth: float, radius: float):
    sdf = df.sdf_knife(point, spine_dim, spine_height,
                       edge_dim, tip_height, depth)
    # dz = df.abs(point[2]) - depth
    return sdf - radius


@df.func
def knife_sdf_grad(point: df.float3, spine_dim: float, spine_height: float, edge_dim: float, tip_height: float, depth: float, radius: float):
    # finite differencing for now (central difference)
    eps = 1e-5
    dx0 = knife_sdf(df.float3(point[0] - eps, point[1], point[2]),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    dx1 = knife_sdf(df.float3(point[0] + eps, point[1], point[2]),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    dy0 = knife_sdf(df.float3(point[0], point[1] - eps, point[2]),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    dy1 = knife_sdf(df.float3(point[0], point[1] + eps, point[2]),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    dz0 = knife_sdf(df.float3(point[0], point[1], point[2] - eps),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    dz1 = knife_sdf(df.float3(point[0], point[1], point[2] + eps),
                    spine_dim, spine_height, edge_dim, tip_height, depth, radius)
    eps2 = 2.0 * eps
    return df.float3((dx1 - dx0) / eps2, (dy1 - dy0) / eps2, (dz1 - dz0) / eps2)


@df.func
def edge_knife_sdf(u: float, a: df.float3, b: df.float3, spine_dim: float, spine_height: float, edge_dim: float, tip_height: float, depth: float,
                   radius: float):
    p = a * u + b * (1.0 - u)
    return knife_sdf(p, spine_dim, spine_height, edge_dim, tip_height, depth, radius)


@df.func
def edge_knife_sdf_grad(u: float,
                        a: df.float3,
                        b: df.float3,
                        spine_dim: float,
                        spine_height: float,
                        edge_dim: float,
                        tip_height: float,
                        depth: float,
                        radius: float):
    eps = 1e-5
    du0 = edge_knife_sdf(u - eps, a, b, spine_dim,
                         spine_height, edge_dim, tip_height, depth, radius)
    du1 = edge_knife_sdf(u + eps, a, b, spine_dim,
                         spine_height, edge_dim, tip_height, depth, radius)
    return (du1 - du0) / (2.0 * eps)


@df.func
def edge_knife_sdf_collision(t: float,
                             xi_shape: df.float3,
                             xj_shape: df.float3,
                             spine_dim: float,
                             spine_height: float,
                             edge_dim: float,
                             tip_height: float,
                             depth: float,
                             radius: float,
                             cut_edge_index: int,
                             cut_edge_contact_dist: df.tensor(float),
                             cut_edge_contact_coord: df.tensor(float),
                             cut_edge_contact_normal: df.tensor(df.float3)):
    # Frank-Wolfe algorithm
    u = 0.5 * t
    k = 1.0
    # for iteration in range(0, 20):
    for iteration in range(0, 20):
        # XXX swap xi and xj because u should be (1-t)
        grad = edge_knife_sdf_grad(
            u, xj_shape, xi_shape, spine_dim, spine_height, edge_dim, tip_height, depth, radius)
        # print(grad)
        # s = 1-t if grad < 0, otherwise s = 0
        s = df.max(df.sign(0.0 - grad), 0.0) * t
        gamma = 2. / (2. + k)
        # print(gamma)
        u = u + gamma * (s - u)
        # print(u)
        k = k + 1.0

    p_shape = xi_shape * (1.0 - u) + xj_shape * u
    # TODO compute distance between end point of cutting edge (at t) and u
    c_dist = knife_sdf(p_shape, spine_dim, spine_height,
                       edge_dim, tip_height, depth, radius)
    n = df.normalize(knife_sdf_grad(p_shape, spine_dim,
                     spine_height, edge_dim, tip_height, depth, radius))
    df.store(cut_edge_contact_dist, cut_edge_index, c_dist)
    df.store(cut_edge_contact_coord, cut_edge_index, 1. - u)
    df.store(cut_edge_contact_normal, cut_edge_index, n)
    return c_dist


@df.func
def knife_contact_force(px: df.float3,
                        pv: df.float3,
                        knife_X_s: df.spatial_transform,
                        knife_v_s: df.spatial_vector,
                        contact_dist: float,
                        contact_normal: df.float3,
                        ke: float,
                        kd: float,
                        kf: float,
                        mu: float,
                        radius: float):

    if contact_dist >= 0.0:
        return df.float3(0.0, 0.0, 0.0)

    rigid_x = df.spatial_transform_get_translation(knife_X_s)
    rigid_r = df.spatial_transform_get_rotation(knife_X_s)
    rigid_v = df.spatial_bottom(knife_v_s)
    rigid_w = df.spatial_top(knife_v_s)

    # moment arm
    r = px - rigid_x
    rhat = df.normalize(r)

    # contact_dist = contact_dist - radius

    # contact point velocity
    rigid_x_to_c = rhat * contact_dist
    # this is rigid velocity cross offset, so it's the velocity of the contact point.
    dpdt = rigid_v + df.cross(rigid_w, rigid_x_to_c)

    # diff = px - closest               # vector from particle to contact point
    # dist = df.dot(diff, diff)          # squared distance

    # n = df.float3(0., 0., 1.)
    contact_normal = df.rotate(rigid_r, contact_normal)
    # flip normal
    contact_normal = df.float3(
        0.0 - contact_normal[0], 0.0 - contact_normal[1], 0.0 - contact_normal[2])
    # n = df.normalize(diff)             # points into the object

    c = df.min(contact_dist, 0.0)
    # print(c)
    #c = df.leaky_min(dot(n, rigid_x)-0.01, 0.0, 0.0)
    # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # df.atomic_sub(tri_f, particle_no, fn)

    # normal force (restitution coefficient * how far inside for ground) (negative)
    fn = c * ke

    vrel = pv - dpdt         # XXX important
    # vrel = df.float3(0.0 - dpdt[0], 0.0 - dpdt[1], 0.0 - dpdt[2])

    # velocity component of rigid in negative normal direction
    vn = df.dot(contact_normal, vrel)
    vt = vrel - contact_normal * vn    # velocity component not in normal direction

    # print(vt[2])

    # contact damping
    # again, negative, into the ground
    fd = 0.0 - df.max(vn, 0.0) * kd * df.step(c)
    f_total = contact_normal * (fn + fd)

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = 0.0 - lower      # workaround because no unary ops yet

    # nx = df.cross(n, df.float3(0.0, 0.0, 1.0))         # basis vectors for tangent
    # nz = df.cross(n, df.float3(1.0, 0.0, 0.0))

    # vx = df.clamp(df.dot(nx * kf, vt), lower, upper)
    # vz = df.clamp(df.dot(nz * kf, vt), lower, upper)

    # ft = (nx * vx + nz * vz) * (0.0 - df.step(c))          # df.float3(vx, 0.0, vz)*df.step(c)

    # print(fn)

    epsilon = 0.0
    # velocity component not in normal direction
    vt_raw = vrel - contact_normal * vn

    # regularize the lateral velocity component to avoid gradient problems when |vt| = 0
    # df.float3(vt_raw[0] + epsilon, vt_raw[1] + epsilon, vt_raw[2] + epsilon)
    vt = vt_raw
    # print(vt)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # ft = df.normalize(vt) * df.min(kf * df.length(vt), 0.0 - mu * fn)
    # f_total = f_total + ft

    # Simplified three-parameter model (Eq. (4.5))
    # Brown "Contact Modelling for Forward Dynamics of Human Motion"
    vt_norm = df.length(vt)
    if vt_norm > 0.0:
        # 10.0 # kf  # XXX kf is the inverse of the transition velocity v_t here
        vvt = vt_norm / kf
        if vvt > 0.0:
            denom = 0.25 * vvt * vvt + 0.75
            mu_static = 0.5
            fr = fn * (mu * df.tanh(4.0 * vvt) +
                       (mu_static - mu) * vvt / (denom * denom))
            ft = df.normalize(vt) * fr
            f_total = f_total + ft
            # print(f_total[1])
            # print(ke)

    return f_total


@df.kernel
def eval_spring_sdf_contacts(
        spring_indices: df.tensor(int),
        edge_indices: df.tensor(int),
        edge_coords: df.tensor(float),
        # number of particles (size of contact_point)
        num_springs: int,
        x: df.tensor(df.float3),                     # position of particles
        v: df.tensor(df.float3),
        knife_id: int,
        knife_X_s: df.tensor(df.spatial_transform),
        knife_v_s: df.tensor(df.spatial_vector),
        knife_params: df.tensor(float),
        sdf_ke: df.tensor(float),
        sdf_kd: df.tensor(float),
        sdf_kf: df.tensor(float),
        sdf_mu: df.tensor(float),
        sdf_radius: float,
        particle_f: df.tensor(df.float3),
        knife_f: df.tensor(df.float3),
        cut_edge_contact_coord: df.tensor(float),
        cut_edge_contact_dist: df.tensor(float),
        cut_edge_contact_normal: df.tensor(df.float3)):

    tid = df.tid()
    spring_no = tid % num_springs      # which spring

    spine_dim = df.load(knife_params, 0)
    spine_height = df.load(knife_params, 1)
    edge_dim = df.load(knife_params, 2)
    tip_height = df.load(knife_params, 3)
    depth = df.load(knife_params, 4)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(sdf_ke, spring_no)    # restitution coefficient
    kd = df.load(sdf_kd, spring_no)    # damping coefficient
    kf = df.load(sdf_kf, spring_no)    # friction coefficient
    mu = df.load(sdf_mu, spring_no)    # coulomb friction

    i = df.load(spring_indices, spring_no * 2 + 0)
    j = df.load(spring_indices, spring_no * 2 + 1)

    knife_X = df.load(knife_X_s, knife_id)
    knife_v = df.load(knife_v_s, knife_id)

    # transform points to shape space
    rigid_x = df.spatial_transform_get_translation(knife_X)
    rigid_r = df.spatial_transform_get_rotation(knife_X)

    l1 = df.load(edge_indices, i * 2 + 0)
    l2 = df.load(edge_indices, i * 2 + 1)
    lx1 = df.load(x, l1)
    lx2 = df.load(x, l2)
    lv1 = df.load(v, l1)
    lv2 = df.load(v, l2)
    lc = df.load(edge_coords, i)

    xi_shape = df.rotate_inv(rigid_r, lx1 - rigid_x)
    xj_shape = df.rotate_inv(rigid_r, lx2 - rigid_x)
    edge_knife_sdf_collision(lc,
                             xi_shape,
                             xj_shape,
                             spine_dim,
                             spine_height,
                             edge_dim,
                             tip_height,
                             depth,
                             sdf_radius,
                             i,
                             cut_edge_contact_dist,
                             cut_edge_contact_coord,
                             cut_edge_contact_normal)
    contact_coord = df.load(cut_edge_contact_coord, i)
    contact_dist = df.load(cut_edge_contact_dist, i)
    contact_normal = df.load(cut_edge_contact_normal, i)
    px = lx1 * (1.0 - contact_coord) + lx2 * contact_coord
    pv = lv1 * (1.0 - contact_coord) + lv2 * contact_coord
    lf = knife_contact_force(
        px, pv, knife_X, knife_v, contact_dist, contact_normal, ke, kd, kf, mu, sdf_radius)
    df.atomic_add(particle_f, l1, lf * contact_coord)
    df.atomic_add(particle_f, l2, lf * (1.0 - contact_coord))
    idx_l = spring_no * 2 + 0
    df.atomic_add(knife_f, idx_l, lf)

    # XXX need to swap coordinates on the side "below" the cut
    r1 = df.load(edge_indices, j * 2 + 1)
    r2 = df.load(edge_indices, j * 2 + 0)
    rx1 = df.load(x, r1)
    rx2 = df.load(x, r2)
    rv1 = df.load(v, r1)
    rv2 = df.load(v, r2)
    rc = df.load(edge_coords, j)
    rc = 1.0 - rc  # XXX invert below the cut

    xi_shape = df.rotate_inv(rigid_r, rx1 - rigid_x)
    xj_shape = df.rotate_inv(rigid_r, rx2 - rigid_x)
    edge_knife_sdf_collision(rc,
                             xi_shape,
                             xj_shape,
                             spine_dim,
                             spine_height,
                             edge_dim,
                             tip_height,
                             depth,
                             sdf_radius,
                             j,
                             cut_edge_contact_dist,
                             cut_edge_contact_coord,
                             cut_edge_contact_normal)
    contact_coord = df.load(cut_edge_contact_coord, j)
    contact_dist = df.load(cut_edge_contact_dist, j)
    contact_normal = df.load(cut_edge_contact_normal, j)
    px = rx1 * (1.0 - contact_coord) + rx2 * contact_coord
    pv = rv1 * (1.0 - contact_coord) + rv2 * contact_coord
    rf = knife_contact_force(
        px, pv, knife_X, knife_v, contact_dist, contact_normal, ke, kd, kf, mu, sdf_radius)
    df.atomic_add(particle_f, r1, rf * contact_coord)
    df.atomic_add(particle_f, r2, rf * (1.0 - contact_coord))
    idx_r = spring_no * 2 + 1
    df.atomic_add(knife_f, idx_r, rf)


@df.func
def cut_edge_tri_collision(cut_edge_coord: float,
                           xi: df.float3,
                           xj: df.float3,
                           tri_0: df.float3,
                           tri_1: df.float3,
                           tri_2: df.float3,
                           sdf_radius: float,
                           cut_edge_index: int,
                           cut_edge_contact_dist: df.tensor(float),
                           cut_edge_contact_coord: df.tensor(float),
                           cut_edge_contact_normal: df.tensor(df.float3)):
    # Möller-Trumbore algorithm
    edge1 = tri_1 - tri_0
    edge2 = tri_2 - tri_0
    dir = xj - xi
    origin = xi
    tri_normal = df.cross(edge1, edge2)
    tri_normal = df.normalize(tri_normal)
    h = df.cross(dir, edge2)
    a = df.dot(edge1, h)
    if (a > 0.0 - 1e-8 and a < 1e-8):
        return 1.0
    f = 1.0 / a
    s = origin - tri_0
    u = f * df.dot(s, h)
    if (u < 0.0 or u > 1.0):
        return 1.0
    q = df.cross(s, edge1)
    v = f * df.dot(dir, q)
    if (v < 0.0 or u + v > 1.0):
        return 1.0
    # intersection on the line
    t = f * df.dot(edge2, q)
    if (t < 1e-8 or t > cut_edge_coord):
        return 1.0

    # measure distance by how far the point at cut_edge_coord pokes into the triangle
    # dist = 0.0 - df.length(dir * (cut_edge_coord - t)) - sdf_radius
    dist = df.dot(dir * (cut_edge_coord - t), tri_normal)
    # compare against existing contact distance
    existing_dist = df.load(cut_edge_contact_dist, cut_edge_index)
    existing_coord = df.load(cut_edge_contact_coord, cut_edge_index)
    if (dist < existing_dist and dist < 0.0 and t < existing_coord):
        df.store(cut_edge_contact_dist, cut_edge_index, dist)
        df.store(cut_edge_contact_coord, cut_edge_index, t)
        df.store(cut_edge_contact_normal, cut_edge_index, tri_normal)
    return 0.0 - sdf_radius


@df.kernel
def eval_spring_tri_collision(
        spring_indices: df.tensor(int),
        edge_indices: df.tensor(int),
        edge_coords: df.tensor(float),
        # number of particles (size of contact_point)
        num_springs: int,
        x: df.tensor(df.float3),                     # position of particles
        sdf_radius: float,
        knife_tri_indices: df.tensor(int),
        knife_tri_vertices: df.tensor(df.float3),
        knife_id: int,
        knife_X_s: df.tensor(df.spatial_transform),
        cut_edge_contact_coord: df.tensor(float),
        cut_edge_contact_dist: df.tensor(float),
        cut_edge_contact_normal: df.tensor(df.float3)):

    tid = df.tid()

    face_no = tid // num_springs       # which triangle
    spring_no = tid % num_springs      # which spring

    i = df.load(spring_indices, spring_no * 2 + 0)
    j = df.load(spring_indices, spring_no * 2 + 1)

    knife_X = df.load(knife_X_s, knife_id)

    a = df.load(knife_tri_indices, face_no * 3 + 0)
    b = df.load(knife_tri_indices, face_no * 3 + 1)
    c = df.load(knife_tri_indices, face_no * 3 + 2)

    xa = df.spatial_transform_point(knife_X, df.load(knife_tri_vertices, a))
    xb = df.spatial_transform_point(knife_X, df.load(knife_tri_vertices, b))
    xc = df.spatial_transform_point(knife_X, df.load(knife_tri_vertices, c))

    l1 = df.load(edge_indices, i * 2 + 0)
    l2 = df.load(edge_indices, i * 2 + 1)
    lx1 = df.load(x, l1)
    lx2 = df.load(x, l2)
    lc = df.load(edge_coords, i)

    cut_edge_tri_collision(lc, lx1, lx2, xa, xb, xc, sdf_radius, i,
                           cut_edge_contact_dist, cut_edge_contact_coord, cut_edge_contact_normal)

    # XXX need to swap coordinates on the side "below" the cut
    r1 = df.load(edge_indices, j * 2 + 1)
    r2 = df.load(edge_indices, j * 2 + 0)
    rx1 = df.load(x, r1)
    rx2 = df.load(x, r2)
    rc = df.load(edge_coords, j)
    rc = 1.0 - rc  # XXX invert below the cut

    cut_edge_tri_collision(rc, rx1, rx2, xa, xb, xc, sdf_radius, j,
                           cut_edge_contact_dist, cut_edge_contact_coord, cut_edge_contact_normal)


@df.kernel
def eval_spring_tri_contacts(
    spring_indices: df.tensor(int),
    edge_indices: df.tensor(int),
    edge_coords: df.tensor(float),
    # number of particles (size of contact_point)
    num_springs: int,
    # position of particles
    x: df.tensor(df.float3),
    v: df.tensor(df.float3),
    knife_id: int,
    knife_X_s: df.tensor(df.spatial_transform),
    knife_v_s: df.tensor(df.spatial_vector),
    cut_edge_contact_coord: df.tensor(float),
    cut_edge_contact_dist: df.tensor(float),
    cut_edge_contact_normal: df.tensor(df.float3),
    sdf_ke: df.tensor(float),
    sdf_kd: df.tensor(float),
    sdf_kf: df.tensor(float),
    sdf_mu: df.tensor(float),
    sdf_radius: df.tensor(float),
    particle_f: df.tensor(df.float3),
    knife_f: df.tensor(df.float3),
):

    tid = df.tid()

    shape_no = tid // num_springs      # which shape
    spring_no = tid % num_springs      # which spring

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(sdf_ke, spring_no)    # restitution coefficient
    kd = df.load(sdf_kd, spring_no)    # damping coefficient
    kf = df.load(sdf_kf, spring_no)    # friction coefficient
    mu = df.load(sdf_mu, spring_no)    # coulomb friction

    radius = df.load(sdf_radius, shape_no)

    i = df.load(spring_indices, spring_no * 2 + 0)
    j = df.load(spring_indices, spring_no * 2 + 1)

    knife_X = df.load(knife_X_s, knife_id)
    knife_v = df.load(knife_v_s, knife_id)

    l1 = df.load(edge_indices, i * 2 + 0)
    l2 = df.load(edge_indices, i * 2 + 1)
    lx1 = df.load(x, l1)
    lx2 = df.load(x, l2)
    lv1 = df.load(v, l1)
    lv2 = df.load(v, l2)

    contact_coord = df.load(cut_edge_contact_coord, i)
    contact_dist = df.load(cut_edge_contact_dist, i)
    contact_normal = df.load(cut_edge_contact_normal, i)
    px = lx1 * (1.0 - contact_coord) + lx2 * contact_coord
    pv = lv1 * (1.0 - contact_coord) + lv2 * contact_coord
    lf = knife_contact_force(
        px, pv, knife_X, knife_v, contact_dist, contact_normal, ke, kd, kf, mu, radius)
    # lu = df.load(cut_spring_contact_coords, i)
    df.atomic_add(particle_f, l1, lf * contact_coord)
    df.atomic_add(particle_f, l2, lf * (1.0 - contact_coord))

    idx_l = spring_no * 2 + 0
    df.atomic_add(knife_f, idx_l, lf)

    # XXX need to swap coordinates on the side "below" the cut
    r1 = df.load(edge_indices, j * 2 + 1)
    r2 = df.load(edge_indices, j * 2 + 0)
    rx1 = df.load(x, r1)
    rx2 = df.load(x, r2)
    rv1 = df.load(v, r1)
    rv2 = df.load(v, r2)
    rc = df.load(edge_coords, j)
    rc = 1.0 - rc  # XXX invert below the cut

    contact_coord = df.load(cut_edge_contact_coord, j)
    contact_dist = df.load(cut_edge_contact_dist, j)
    contact_normal = df.load(cut_edge_contact_normal, j)
    px = rx1 * (1.0 - contact_coord) + rx2 * contact_coord
    pv = rv1 * (1.0 - contact_coord) + rv2 * contact_coord
    rf = knife_contact_force(
        px, pv, knife_X, knife_v, contact_dist, contact_normal, ke, kd, kf, mu, radius)
    df.atomic_add(particle_f, r1, rf * contact_coord)
    df.atomic_add(particle_f, r2, rf * (1.0 - contact_coord))

    idx_r = spring_no * 2 + 1
    df.atomic_add(knife_f, idx_r, rf)


@df.kernel
def eval_cut_springs(
        x: df.tensor(df.float3),
        v: df.tensor(df.float3),
        edge_indices: df.tensor(int),
        edge_coords: df.tensor(float),
        spring_indices: df.tensor(int),
        spring_rest_lengths: df.tensor(float),
        spring_stiffness: df.tensor(float),
        spring_damping: df.tensor(float),
        spring_normal: df.tensor(df.float3),
        spring_contact_stiffness: df.tensor(float),
        spring_softness: df.tensor(float),
        knife_f: df.tensor(df.float3),
        f: df.tensor(df.float3),
        spring_stiffness_out: df.tensor(float),
):

    tid = df.tid()

    i = df.load(spring_indices, tid * 2 + 0)
    j = df.load(spring_indices, tid * 2 + 1)

    ke = df.load(spring_stiffness, tid)
    kd = df.load(spring_damping, tid)
    rest = df.load(spring_rest_lengths, tid)
    softness = df.load(spring_softness, tid)

    l1 = df.load(edge_indices, i * 2 + 0)
    l2 = df.load(edge_indices, i * 2 + 1)
    lx1 = df.load(x, l1)
    lx2 = df.load(x, l2)
    lv1 = df.load(v, l1)
    lv2 = df.load(v, l2)
    lc = df.load(edge_coords, i)
    xi = lx1 * (1.0 - lc) + lx2 * lc
    vi = lv1 * (1.0 - lc) + lv2 * lc

    r1 = df.load(edge_indices, j * 2 + 0)
    r2 = df.load(edge_indices, j * 2 + 1)
    rx1 = df.load(x, r1)
    rx2 = df.load(x, r2)
    rv1 = df.load(v, r1)
    rv2 = df.load(v, r2)
    rc = df.load(edge_coords, j)
    xj = rx1 * (1.0 - rc) + rx2 * rc
    vj = rv1 * (1.0 - rc) + rv2 * rc

    xij = xi - xj
    l = df.length(xij)

    # compute spring displacement
    displacement = l - rest
    if l > 0.0:
        vij = vi - vj

        # compute contact forces due to self-collision
        normal = df.load(spring_normal, tid)
        contact_ke = df.load(spring_contact_stiffness, tid)

        diff = xi - xj
        # contact_c = df.leaky_min(df.dot(normal, diff)-rest, 0.0, 0.0)
        contact_c = df.min(df.dot(normal, diff)-rest, 0.0)

        contact_fn = normal * (contact_c * contact_ke)

        df.atomic_sub(f, l1, contact_fn * (1.0 - lc))
        df.atomic_sub(f, l2, contact_fn * lc)
        df.atomic_add(f, r1, contact_fn * (1.0 - rc))
        df.atomic_add(f, r2, contact_fn * rc)

        l_inv = 1.0 / l

        # normalized spring direction
        dir = xij * l_inv

        dcdt = df.dot(dir, vij)

        # damping based on relative velocity.
        fs = dir * (ke * displacement + kd * dcdt)

        df.atomic_sub(f, l1, fs * (1.0 - lc))
        df.atomic_sub(f, l2, fs * lc)
        df.atomic_add(f, r1, fs * (1.0 - rc))
        df.atomic_add(f, r2, fs * rc)

    # compute spring displacement for the damage model
    displacement = df.max(0., displacement)

    # only damage springs when they are in contact
    idx_l = tid * 2 + 0
    idx_r = tid * 2 + 1
    knife_f_l = df.load(knife_f, idx_l)
    knife_f_r = df.load(knife_f, idx_r)
    incontact = df.nonzero(df.length(knife_f_l) + df.length(knife_f_r))

    # only update spring stiffness
    new_spring_stiffness = df.max(
        0.0, ke - incontact * displacement * softness)
    df.store(spring_stiffness_out, tid, new_spring_stiffness)


@df.kernel
def eval_soft_contacts(
        num_particles: int,
        particle_x: df.tensor(df.float3),
        particle_v: df.tensor(df.float3),
        body_X_sc: df.tensor(df.spatial_transform),
        body_v_sc: df.tensor(df.spatial_vector),
        shape_X_co: df.tensor(df.spatial_transform),
        shape_body: df.tensor(int),
        shape_geo_type: df.tensor(int),
        shape_geo_src: df.tensor(int),
        shape_geo_scale: df.tensor(df.float3),
        shape_materials: df.tensor(float),
        ke: float,
        kd: float,
        kf: float,
        mu: float,
        # outputs
        particle_f: df.tensor(df.float3),
        body_f: df.tensor(df.spatial_vector)):

    tid = df.tid()

    shape_index = tid // num_particles           # which shape
    particle_index = tid % num_particles         # which particle
    rigid_index = df.load(shape_body, shape_index)

    px = df.load(particle_x, particle_index)
    pv = df.load(particle_v, particle_index)

    #center = float3(0.0, 0.5, 0.0)
    #radius = 0.25
    #margin = 0.01

    # sphere collider
    # c = df.min(sphere_sdf(center, radius, x0)-margin, 0.0)
    # n = sphere_sdf_grad(center, radius, x0)

    # box collider
    #c = df.min(box_sdf(df.float3(radius, radius, radius), x0-center)-margin, 0.0)
    #n = box_sdf_grad(df.float3(radius, radius, radius), x0-center)

    X_sc = df.spatial_transform_identity()
    if (rigid_index >= 0):
        X_sc = df.load(body_X_sc, rigid_index)

    X_co = df.load(shape_X_co, shape_index)

    X_so = df.spatial_transform_multiply(X_sc, X_co)
    X_os = df.spatial_transform_inverse(X_so)

    # transform particle position to shape local space
    x_local = df.spatial_transform_point(X_os, px)

    # geo description
    geo_type = df.load(shape_geo_type, shape_index)
    geo_scale = df.load(shape_geo_scale, shape_index)

    margin = 0.01

    # evaluate shape sdf
    c = 0.0
    n = df.float3(0.0, 0.0, 0.0)

    # GEO_SPHERE (0)
    if (geo_type == 0):
        c = df.min(sphere_sdf(df.float3(0.0, 0.0, 0.0),
                   geo_scale[0], x_local) - margin, 0.0)
        n = df.spatial_transform_vector(X_so, sphere_sdf_grad(
            df.float3(0.0, 0.0, 0.0), geo_scale[0], x_local))

    # GEO_BOX (1)
    if (geo_type == 1):
        c = df.min(box_sdf(geo_scale, x_local) - margin, 0.0)
        n = df.spatial_transform_vector(X_so, box_sdf_grad(geo_scale, x_local))

    # GEO_CAPSULE (2)
    if (geo_type == 2):
        c = df.min(capsule_sdf(
            geo_scale[0], geo_scale[1], x_local) - margin, 0.0)
        n = df.spatial_transform_vector(
            X_so, capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local))

    # rigid velocity
    rigid_v_s = df.spatial_vector()
    if (rigid_index >= 0):
        rigid_v_s = df.load(body_v_sc, rigid_index)

    rigid_w = df.spatial_top(rigid_v_s)
    rigid_v = df.spatial_bottom(rigid_v_s)

    # compute the body velocity at the particle position
    bv = rigid_v + df.cross(rigid_w, px)

    # relative velocity
    v = pv - bv

    # decompose relative velocity
    vn = dot(n, v)
    vt = v - n * vn

    # contact elastic
    fn = n * c * ke

    # contact damping
    fd = n * df.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = fn + (fd + ft) * df.step(c)
    t_total = df.cross(px, f_total)

    df.atomic_sub(particle_f, particle_index, f_total)

    if (rigid_index >= 0):
        df.atomic_sub(body_f, rigid_index, df.spatial_vector(t_total, f_total))


@df.kernel
def eval_rigid_contacts(rigid_x: df.tensor(df.float3),
                        rigid_r: df.tensor(df.quat),
                        rigid_v: df.tensor(df.float3),
                        rigid_w: df.tensor(df.float3),
                        contact_body: df.tensor(int),
                        contact_point: df.tensor(df.float3),
                        contact_dist: df.tensor(float),
                        contact_mat: df.tensor(int),
                        materials: df.tensor(float),
                        rigid_f: df.tensor(df.float3),
                        rigid_t: df.tensor(df.float3)):

    tid = df.tid()

    c_body = df.load(contact_body, tid)
    c_point = df.load(contact_point, tid)
    c_dist = df.load(contact_dist, tid)
    c_mat = df.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = df.load(rigid_x, c_body)      # position of colliding body
    r0 = df.load(rigid_r, c_body)      # orientation of colliding body

    v0 = df.load(rigid_v, c_body)
    w0 = df.load(rigid_w, c_body)

    n = float3(0.0, 1.0, 0.0)

    # transform point to world space
    # add on 'thickness' of shape, e.g.: radius of sphere/capsule
    p = x0 + df.rotate(r0, c_point) - n * c_dist
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = p - x0     # basically just c_point in the new coordinates

    # contact point velocity
    # this is rigid velocity cross offset, so it's the velocity of the contact point.
    dpdt = v0 + df.cross(w0, r)

    # check ground contact
    c = df.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    # normal force (restitution coefficient * how far inside for ground)
    fn = c * ke

    # contact damping
    # again, velocity into the ground, negative
    fd = df.min(vn, 0.0) * kd * df.step(c)

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = df.clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = df.clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz) * df.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = df.cross(r, f_total)

    df.atomic_sub(rigid_f, c_body, f_total)
    df.atomic_sub(rigid_t, c_body, t_total)


# # Frank & Park definition 3.20, pg 100
@df.func
def spatial_transform_twist(t: df.spatial_transform, x: df.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    w = rotate(q, w)
    v = rotate(q, v) + cross(p, w)

    return spatial_vector(w, v)


@df.func
def spatial_transform_wrench(t: df.spatial_transform, x: df.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    v = rotate(q, v)
    w = rotate(q, w) + cross(p, v)

    return spatial_vector(w, v)


@df.func
def spatial_transform_inverse(t: df.spatial_transform):

    p = spatial_transform_get_translation(t)
    q = spatial_transform_get_rotation(t)

    q_inv = inverse(q)
    return spatial_transform(rotate(q_inv, p) * (0.0 - 1.0), q_inv)


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@df.func
def spatial_transform_inertia(t: df.spatial_transform, I: df.spatial_matrix):

    t_inv = spatial_transform_inverse(t)

    q = spatial_transform_get_rotation(t_inv)
    p = spatial_transform_get_translation(t_inv)

    r1 = rotate(q, float3(1.0, 0.0, 0.0))
    r2 = rotate(q, float3(0.0, 1.0, 0.0))
    r3 = rotate(q, float3(0.0, 0.0, 1.0))

    R = mat33(r1, r2, r3)
    S = mul(skew(p), R)

    T = spatial_adjoint(R, S)

    return mul(mul(transpose(T), I), T)


@df.kernel
def eval_rigid_contacts_art(body_X_s: df.tensor(df.spatial_transform),
                            body_v_s: df.tensor(df.spatial_vector),
                            contact_body: df.tensor(int),
                            contact_point: df.tensor(df.float3),
                            contact_dist: df.tensor(float),
                            contact_mat: df.tensor(int),
                            materials: df.tensor(float),
                            body_f_s: df.tensor(df.spatial_vector)):

    tid = df.tid()

    c_body = df.load(contact_body, tid)
    c_point = df.load(contact_point, tid)
    c_dist = df.load(contact_dist, tid)
    c_mat = df.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = df.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = df.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = df.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = df.load(materials, c_mat * 4 + 3)       # coulomb friction

    X_s = df.load(body_X_s, c_body)    # position of colliding body
    v_s = df.load(body_v_s, c_body)    # orientation of colliding body

    n = float3(0.0, 1.0, 0.0)

    # transform point to world space
    # add on 'thickness' of shape, e.g.: radius of sphere/capsule
    p = df.spatial_transform_point(X_s, c_point) - n * c_dist

    w = df.spatial_top(v_s)
    v = df.spatial_bottom(v_s)

    # contact point velocity
    dpdt = v + df.cross(w, p)

    # check ground contact
    c = df.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    # normal force (restitution coefficient * how far inside for ground)
    fn = c * ke

    # contact damping
    # again, velocity into the ground, negative
    fd = df.min(vn, 0.0) * kd * df.step(c)

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = df.clamp(dot(float3(kf, 0.0, 0.0), vt), lower, upper)
    vz = df.clamp(dot(float3(0.0, 0.0, kf), vt), lower, upper)

    ft = df.float3(vx, 0.0, vz) * df.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = df.normalize(vt)*df.min(kf*df.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = df.cross(p, f_total)

    df.atomic_add(body_f_s, c_body, df.spatial_vector(t_total, f_total))


@df.func
def compute_muscle_force(i: int,
                         body_X_s: df.tensor(df.spatial_transform),
                         body_v_s: df.tensor(df.spatial_vector),
                         muscle_links: df.tensor(int),
                         muscle_points: df.tensor(df.float3),
                         muscle_activation: float,
                         body_f_s: df.tensor(df.spatial_vector)):

    link_0 = df.load(muscle_links, i)
    link_1 = df.load(muscle_links, i + 1)

    if (link_0 == link_1):
        return 0

    r_0 = df.load(muscle_points, i)
    r_1 = df.load(muscle_points, i + 1)

    xform_0 = df.load(body_X_s, link_0)
    xform_1 = df.load(body_X_s, link_1)

    pos_0 = df.spatial_transform_point(xform_0, r_0)
    pos_1 = df.spatial_transform_point(xform_1, r_1)

    n = df.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    df.atomic_sub(body_f_s, link_0, df.spatial_vector(df.cross(pos_0, f), f))
    df.atomic_add(body_f_s, link_1, df.spatial_vector(df.cross(pos_1, f), f))

    return 0


@df.kernel
def eval_muscles(
        body_X_s: df.tensor(df.spatial_transform),
        body_v_s: df.tensor(df.spatial_vector),
        muscle_start: df.tensor(int),
        muscle_params: df.tensor(float),
        muscle_links: df.tensor(int),
        muscle_points: df.tensor(df.float3),
        muscle_activation: df.tensor(float),
    # output
        body_f_s: df.tensor(df.spatial_vector)):

    tid = df.tid()

    m_start = df.load(muscle_start, tid)
    m_end = df.load(muscle_start, tid + 1) - 1

    activation = df.load(muscle_activation, tid)

    for i in range(m_start, m_end):
        compute_muscle_force(i, body_X_s, body_v_s,
                             muscle_links, muscle_points, activation, body_f_s)


# compute transform across a joint
@df.func
def jcalc_transform(type: int, axis: df.float3, joint_q: df.tensor(float), start: int):

    # prismatic
    if (type == 0):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(axis * q, quat_identity())
        return X_jc

    # revolute
    if (type == 1):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(
            float3(0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if (type == 2):

        qx = df.load(joint_q, start + 0)
        qy = df.load(joint_q, start + 1)
        qz = df.load(joint_q, start + 2)
        qw = df.load(joint_q, start + 3)

        X_jc = spatial_transform(float3(0.0, 0.0, 0.0), quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if (type == 3):

        X_jc = spatial_transform_identity()
        return X_jc

    # free
    if (type == 4):

        px = df.load(joint_q, start + 0)
        py = df.load(joint_q, start + 1)
        pz = df.load(joint_q, start + 2)

        qx = df.load(joint_q, start + 3)
        qy = df.load(joint_q, start + 4)
        qz = df.load(joint_q, start + 5)
        qw = df.load(joint_q, start + 6)

        X_jc = spatial_transform(float3(px, py, pz), quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return spatial_transform_identity()


# compute motion subspace and velocity for a joint
@df.func
def jcalc_motion(type: int, axis: df.float3, X_sc: df.spatial_transform, joint_S_s: df.tensor(df.spatial_vector), joint_qd: df.tensor(float), joint_start: int):

    # prismatic
    if (type == 0):

        S_s = df.spatial_transform_twist(
            X_sc, spatial_vector(float3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # revolute
    if (type == 1):

        S_s = df.spatial_transform_twist(
            X_sc, spatial_vector(axis, float3(0.0, 0.0, 0.0)))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # ball
    if (type == 2):

        w = float3(df.load(joint_qd, joint_start + 0), df.load(joint_qd,
                   joint_start + 1), df.load(joint_qd, joint_start + 2))

        S_0 = df.spatial_transform_twist(
            X_sc, spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = df.spatial_transform_twist(
            X_sc, spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = df.spatial_transform_twist(
            X_sc, spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        df.store(joint_S_s, joint_start + 0, S_0)
        df.store(joint_S_s, joint_start + 1, S_1)
        df.store(joint_S_s, joint_start + 2, S_2)

        return S_0 * w[0] + S_1 * w[1] + S_2 * w[2]

    # fixed
    if (type == 3):
        return spatial_vector()

    # free
    if (type == 4):

        v_j_s = spatial_vector(df.load(joint_qd, joint_start + 0),
                               df.load(joint_qd, joint_start + 1),
                               df.load(joint_qd, joint_start + 2),
                               df.load(joint_qd, joint_start + 3),
                               df.load(joint_qd, joint_start + 4),
                               df.load(joint_qd, joint_start + 5))

        # write motion subspace
        df.store(joint_S_s, joint_start + 0,
                 spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        df.store(joint_S_s, joint_start + 1,
                 spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        df.store(joint_S_s, joint_start + 2,
                 spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        df.store(joint_S_s, joint_start + 3,
                 spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        df.store(joint_S_s, joint_start + 4,
                 spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        df.store(joint_S_s, joint_start + 5,
                 spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    # default case
    return spatial_vector()


# # compute the velocity across a joint
# #@df.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = spatial_vector()
#         return v_j_s

#     # free
#     if (type == 3):
#         v_j_s =  S_s[start+0]*joint_qd[start+0]
#         v_j_s += S_s[start+1]*joint_qd[start+1]
#         v_j_s += S_s[start+2]*joint_qd[start+2]
#         v_j_s += S_s[start+3]*joint_qd[start+3]
#         v_j_s += S_s[start+4]*joint_qd[start+4]
#         v_j_s += S_s[start+5]*joint_qd[start+5]
#         return v_j_s


# computes joint space forces/torques in tau
@df.func
def jcalc_tau(type: int,
              target_k_e: float,
              target_k_d: float,
              limit_k_e: float,
              limit_k_d: float,
              joint_S_s: df.tensor(spatial_vector),
              joint_q: df.tensor(float),
              joint_qd: df.tensor(float),
              joint_act: df.tensor(float),
              joint_target: df.tensor(float),
              joint_limit_lower: df.tensor(float),
              joint_limit_upper: df.tensor(float),
              coord_start: int,
              dof_start: int,
              body_f_s: spatial_vector,
              tau: df.tensor(float)):

    # prismatic / revolute
    if (type == 0 or type == 1):
        S_s = df.load(joint_S_s, dof_start)

        q = df.load(joint_q, coord_start)
        qd = df.load(joint_qd, dof_start)
        act = df.load(joint_act, dof_start)

        target = df.load(joint_target, coord_start)
        lower = df.load(joint_limit_lower, coord_start)
        upper = df.load(joint_limit_upper, coord_start)

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if (q < lower):
            limit_f = limit_k_e * (lower - q) - limit_k_d * min(qd, 0.0)

        if (q > upper):
            limit_f = limit_k_e * (upper - q) - limit_k_d * max(qd, 0.0)

        # total torque / force on the joint
        t = 0.0 - spatial_dot(S_s, body_f_s) - target_k_e * \
            (q - target) - target_k_d * qd + act + limit_f

        df.store(tau, dof_start, t)

    # ball
    if (type == 2):

        # elastic term.. this is proportional to the
        # imaginary part of the relative quaternion
        r_j = float3(df.load(joint_q, coord_start + 0), df.load(joint_q,
                     coord_start + 1), df.load(joint_q, coord_start + 2))

        # angular velocity for damping
        w_j = float3(df.load(joint_qd, dof_start + 0), df.load(joint_qd,
                     dof_start + 1), df.load(joint_qd, dof_start + 2))

        for i in range(0, 3):
            S_s = df.load(joint_S_s, dof_start + i)

            w = w_j[i]
            r = r_j[i]

            df.store(tau, dof_start + i, 0.0 - spatial_dot(S_s,
                     body_f_s) - w * target_k_d - r * target_k_e)

    # fixed
    # if (type == 3)
    #    pass

    # free
    if (type == 4):

        for i in range(0, 6):
            S_s = df.load(joint_S_s, dof_start + i)
            df.store(tau, dof_start + i, 0.0 - spatial_dot(S_s, body_f_s))

    return 0


@df.func
def jcalc_integrate(type: int,
                    joint_q: df.tensor(float),
                    joint_qd: df.tensor(float),
                    joint_qdd: df.tensor(float),
                    coord_start: int,
                    dof_start: int,
                    dt: float,
                    joint_q_new: df.tensor(float),
                    joint_qd_new: df.tensor(float)):

    # prismatic / revolute
    if (type == 0 or type == 1):

        qdd = df.load(joint_qdd, dof_start)
        qd = df.load(joint_qd, dof_start)
        q = df.load(joint_q, coord_start)

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        df.store(joint_qd_new, dof_start, qd_new)
        df.store(joint_q_new, coord_start, q_new)

    # ball
    if (type == 2):

        m_j = float3(df.load(joint_qdd, dof_start + 0), df.load(joint_qdd,
                     dof_start + 1), df.load(joint_qdd, dof_start + 2))

        w_j = float3(df.load(joint_qd, dof_start + 0), df.load(joint_qd,
                     dof_start + 1), df.load(joint_qd, dof_start + 2))

        r_j = quat(df.load(joint_q, coord_start + 0), df.load(joint_q, coord_start + 1),
                   df.load(joint_q, coord_start + 2), df.load(joint_q, coord_start + 3))

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = mul(quat(w_j_new, 0.0), r_j) * 0.5

        # new orientation (normalized)
        r_j_new = normalize(r_j + drdt_j * dt)

        # update joint coords
        df.store(joint_q_new, coord_start + 0, r_j_new[0])
        df.store(joint_q_new, coord_start + 1, r_j_new[1])
        df.store(joint_q_new, coord_start + 2, r_j_new[2])
        df.store(joint_q_new, coord_start + 3, r_j_new[3])

        # update joint vel
        df.store(joint_qd_new, dof_start + 0, w_j_new[0])
        df.store(joint_qd_new, dof_start + 1, w_j_new[1])
        df.store(joint_qd_new, dof_start + 2, w_j_new[2])

    # fixed joint
    # if (type == 3)
    #    pass

    # free joint
    if (type == 4):

        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = float3(df.load(joint_qdd, dof_start + 0), df.load(joint_qdd,
                     dof_start + 1), df.load(joint_qdd, dof_start + 2))

        a_s = float3(df.load(joint_qdd, dof_start + 3), df.load(joint_qdd,
                     dof_start + 4), df.load(joint_qdd, dof_start + 5))

        # angular and linear velocity
        w_s = float3(df.load(joint_qd, dof_start + 0), df.load(joint_qd,
                     dof_start + 1), df.load(joint_qd, dof_start + 2))

        v_s = float3(df.load(joint_qd, dof_start + 3), df.load(joint_qd,
                     dof_start + 4), df.load(joint_qd, dof_start + 5))

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        # translation of origin
        p_s = float3(df.load(joint_q, coord_start + 0), df.load(joint_q,
                     coord_start + 1), df.load(joint_q, coord_start + 2))

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + cross(w_s, p_s)

        # quat and quat derivative
        r_s = quat(df.load(joint_q, coord_start + 3), df.load(joint_q, coord_start + 4),
                   df.load(joint_q, coord_start + 5), df.load(joint_q, coord_start + 6))

        drdt_s = mul(quat(w_s, 0.0), r_s) * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = normalize(r_s + drdt_s * dt)

        # update transform
        df.store(joint_q_new, coord_start + 0, p_s_new[0])
        df.store(joint_q_new, coord_start + 1, p_s_new[1])
        df.store(joint_q_new, coord_start + 2, p_s_new[2])

        df.store(joint_q_new, coord_start + 3, r_s_new[0])
        df.store(joint_q_new, coord_start + 4, r_s_new[1])
        df.store(joint_q_new, coord_start + 5, r_s_new[2])
        df.store(joint_q_new, coord_start + 6, r_s_new[3])

        # update joint_twist
        df.store(joint_qd_new, dof_start + 0, w_s[0])
        df.store(joint_qd_new, dof_start + 1, w_s[1])
        df.store(joint_qd_new, dof_start + 2, w_s[2])
        df.store(joint_qd_new, dof_start + 3, v_s[0])
        df.store(joint_qd_new, dof_start + 4, v_s[1])
        df.store(joint_qd_new, dof_start + 5, v_s[2])

    return 0


@df.func
def compute_link_transform(i: int,
                           joint_type: df.tensor(int),
                           joint_parent: df.tensor(int),
                           joint_q_start: df.tensor(int),
                           joint_qd_start: df.tensor(int),
                           joint_q: df.tensor(float),
                           joint_X_pj: df.tensor(df.spatial_transform),
                           joint_X_cm: df.tensor(df.spatial_transform),
                           joint_axis: df.tensor(df.float3),
                           body_X_sc: df.tensor(df.spatial_transform),
                           body_X_sm: df.tensor(df.spatial_transform)):

    # parent transform
    parent = load(joint_parent, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    type = load(joint_type, i)
    axis = load(joint_axis, i)
    coord_start = load(joint_q_start, i)
    dof_start = load(joint_qd_start, i)

    # compute transform across joint
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = load(joint_X_pj, i)
    X_sc = spatial_transform_multiply(
        X_sp, spatial_transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = load(joint_X_cm, i)
    X_sm = spatial_transform_multiply(X_sc, X_cm)

    # store geometry transforms
    store(body_X_sc, i, X_sc)
    store(body_X_sm, i, X_sm)

    return 0


@df.kernel
def eval_rigid_fk(articulation_start: df.tensor(int),
                  joint_type: df.tensor(int),
                  joint_parent: df.tensor(int),
                  joint_q_start: df.tensor(int),
                  joint_qd_start: df.tensor(int),
                  joint_q: df.tensor(float),
                  joint_X_pj: df.tensor(df.spatial_transform),
                  joint_X_cm: df.tensor(df.spatial_transform),
                  joint_axis: df.tensor(df.float3),
                  body_X_sc: df.tensor(df.spatial_transform),
                  body_X_sm: df.tensor(df.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_start, index + 1)

    for i in range(start, end):
        compute_link_transform(i, joint_type, joint_parent, joint_q_start, joint_qd_start,
                               joint_q, joint_X_pj, joint_X_cm, joint_axis, body_X_sc, body_X_sm)


@df.func
def compute_link_velocity(
        i: int,
        joint_type: df.tensor(int),
        joint_parent: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_qd: df.tensor(float),
        joint_axis: df.tensor(df.float3),
        body_I_m: df.tensor(df.spatial_matrix),
        body_X_sc: df.tensor(df.spatial_transform),
        body_X_sm: df.tensor(df.spatial_transform),
        joint_X_pj: df.tensor(df.spatial_transform),
        gravity: df.tensor(df.float3),
        # outputs
        joint_S_s: df.tensor(df.spatial_vector),
        body_I_s: df.tensor(df.spatial_matrix),
        body_v_s: df.tensor(df.spatial_vector),
        body_f_s: df.tensor(df.spatial_vector),
        body_a_s: df.tensor(df.spatial_vector)):

    type = df.load(joint_type, i)
    axis = df.load(joint_axis, i)
    parent = df.load(joint_parent, i)
    dof_start = df.load(joint_qd_start, i)

    X_sc = df.load(body_X_sc, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    X_pj = load(joint_X_pj, i)
    X_sj = spatial_transform_multiply(X_sp, X_pj)

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sj, joint_S_s, joint_qd, dof_start)

    # parent velocity
    v_parent_s = spatial_vector()
    a_parent_s = spatial_vector()

    if (parent >= 0):
        v_parent_s = df.load(body_v_s, parent)
        a_parent_s = df.load(body_a_s, parent)

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    # + self.joint_S_s[i]*self.joint_qdd[i]
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)

    # compute body forces
    X_sm = df.load(body_X_sm, i)
    I_m = df.load(body_I_m, i)

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    g = df.load(gravity, 0)

    m = I_m[3, 3]

    f_g_m = spatial_vector(float3(), g) * m
    f_g_s = spatial_transform_wrench(spatial_transform(
        spatial_transform_get_translation(X_sm), quat_identity()), f_g_m)

    #f_ext_s = df.load(body_f_s, i) + f_g_s

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = df.mul(I_s, a_s) + spatial_cross_dual(v_s, df.mul(I_s, v_s))

    df.store(body_v_s, i, v_s)
    df.store(body_a_s, i, a_s)
    df.store(body_f_s, i, f_b_s - f_g_s)
    df.store(body_I_s, i, I_s)

    return 0


@df.func
def compute_link_tau(
        offset: int,
        joint_end: int,
        joint_type: df.tensor(int),
        joint_parent: df.tensor(int),
        joint_q_start: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_q: df.tensor(float),
        joint_qd: df.tensor(float),
        joint_act: df.tensor(float),
        joint_target: df.tensor(float),
        joint_target_ke: df.tensor(float),
        joint_target_kd: df.tensor(float),
        joint_limit_lower: df.tensor(float),
        joint_limit_upper: df.tensor(float),
        joint_limit_ke: df.tensor(float),
        joint_limit_kd: df.tensor(float),
        joint_S_s: df.tensor(df.spatial_vector),
        body_fb_s: df.tensor(df.spatial_vector),
        # outputs
        body_ft_s: df.tensor(df.spatial_vector),
        tau: df.tensor(float)):

    # for backwards traversal
    i = joint_end - offset - 1

    type = df.load(joint_type, i)
    parent = df.load(joint_parent, i)
    dof_start = df.load(joint_qd_start, i)
    coord_start = df.load(joint_q_start, i)

    target_k_e = df.load(joint_target_ke, i)
    target_k_d = df.load(joint_target_kd, i)

    limit_k_e = df.load(joint_limit_ke, i)
    limit_k_d = df.load(joint_limit_kd, i)

    # total forces on body
    f_b_s = df.load(body_fb_s, i)
    f_t_s = df.load(body_ft_s, i)

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(type,
              target_k_e,
              target_k_d,
              limit_k_e,
              limit_k_d,
              joint_S_s,
              joint_q,
              joint_qd,
              joint_act,
              joint_target,
              joint_limit_lower,
              joint_limit_upper,
              coord_start,
              dof_start,
              f_s,
              tau)

    # update parent forces, todo: check that this is valid for the backwards pass
    if (parent >= 0):
        df.atomic_add(body_ft_s, parent, f_s)

    return 0


@df.kernel
def eval_rigid_id(
        articulation_start: df.tensor(int),
        joint_type: df.tensor(int),
        joint_parent: df.tensor(int),
        joint_q_start: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_q: df.tensor(float),
        joint_qd: df.tensor(float),
        joint_axis: df.tensor(df.float3),
        joint_target_ke: df.tensor(float),
        joint_target_kd: df.tensor(float),
        body_I_m: df.tensor(df.spatial_matrix),
        body_X_sc: df.tensor(df.spatial_transform),
        body_X_sm: df.tensor(df.spatial_transform),
        joint_X_pj: df.tensor(df.spatial_transform),
        gravity: df.tensor(df.float3),
    # outputs
        joint_S_s: df.tensor(df.spatial_vector),
        body_I_s: df.tensor(df.spatial_matrix),
        body_v_s: df.tensor(df.spatial_vector),
        body_f_s: df.tensor(df.spatial_vector),
        body_a_s: df.tensor(df.spatial_vector)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_start, index + 1)
    count = end - start

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(i,
                              joint_type,
                              joint_parent,
                              joint_qd_start,
                              joint_qd,
                              joint_axis,
                              body_I_m,
                              body_X_sc,
                              body_X_sm,
                              joint_X_pj,
                              gravity,
                              joint_S_s,
                              body_I_s,
                              body_v_s,
                              body_f_s,
                              body_a_s)


@df.kernel
def eval_rigid_tau(
        articulation_start: df.tensor(int),
        joint_type: df.tensor(int),
        joint_parent: df.tensor(int),
        joint_q_start: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_q: df.tensor(float),
        joint_qd: df.tensor(float),
        joint_act: df.tensor(float),
        joint_target: df.tensor(float),
        joint_target_ke: df.tensor(float),
        joint_target_kd: df.tensor(float),
        joint_limit_lower: df.tensor(float),
        joint_limit_upper: df.tensor(float),
        joint_limit_ke: df.tensor(float),
        joint_limit_kd: df.tensor(float),
        joint_axis: df.tensor(df.float3),
        joint_S_s: df.tensor(df.spatial_vector),
        body_fb_s: df.tensor(df.spatial_vector),
    # outputs
        body_ft_s: df.tensor(df.spatial_vector),
        tau: df.tensor(float)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_start, index + 1)
    count = end - start

    # compute joint forces
    for i in range(0, count):
        compute_link_tau(i,
                         end,
                         joint_type,
                         joint_parent,
                         joint_q_start,
                         joint_qd_start,
                         joint_q,
                         joint_qd,
                         joint_act,
                         joint_target,
                         joint_target_ke,
                         joint_target_kd,
                         joint_limit_lower,
                         joint_limit_upper,
                         joint_limit_ke,
                         joint_limit_kd,
                         joint_S_s,
                         body_fb_s,
                         body_ft_s,
                         tau)


@df.kernel
def eval_rigid_jacobian(
        articulation_start: df.tensor(int),
        articulation_J_start: df.tensor(int),
        joint_parent: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_S_s: df.tensor(spatial_vector),
    # outputs
        J: df.tensor(float)):

    # one thread per-articulation
    index = tid()

    joint_start = df.load(articulation_start, index)
    joint_end = df.load(articulation_start, index + 1)
    joint_count = joint_end - joint_start

    J_offset = df.load(articulation_J_start, index)

    # in spatial.h
    spatial_jacobian(joint_S_s, joint_parent, joint_qd_start,
                     joint_start, joint_count, J_offset, J)


@df.kernel
def eval_rigid_mass(
        articulation_start: df.tensor(int),
        articulation_M_start: df.tensor(int),
        body_I_s: df.tensor(spatial_matrix),
    # outputs
        M: df.tensor(float)):

    # one thread per-articulation
    index = tid()

    joint_start = df.load(articulation_start, index)
    joint_end = df.load(articulation_start, index + 1)
    joint_count = joint_end - joint_start

    M_offset = df.load(articulation_M_start, index)

    # in spatial.h
    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@df.kernel
def eval_dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: df.tensor(float), B: df.tensor(float), C: df.tensor(float)):
    dense_gemm(m, n, p, t1, t2, A, B, C)


@df.kernel
def eval_dense_gemm_batched(m: df.tensor(int),
                            n: df.tensor(int),
                            p: df.tensor(int),
                            t1: int,
                            t2: int,
                            A_start: df.tensor(int),
                            B_start: df.tensor(int),
                            C_start: df.tensor(int),
                            A: df.tensor(float),
                            B: df.tensor(float),
                            C: df.tensor(float)):
    dense_gemm_batched(m, n, p, t1, t2, A_start, B_start, C_start, A, B, C)


@df.kernel
def eval_dense_cholesky(n: int, A: df.tensor(float), regularization: df.tensor(float), L: df.tensor(float)):
    dense_chol(n, A, regularization, L)


@df.kernel
def eval_dense_cholesky_batched(A_start: df.tensor(int), A_dim: df.tensor(int), A: df.tensor(float), regularization: df.tensor(float), L: df.tensor(float)):
    dense_chol_batched(A_start, A_dim, A, regularization, L)


@df.kernel
def eval_dense_subs(n: int, L: df.tensor(float), b: df.tensor(float), x: df.tensor(float)):
    dense_subs(n, L, b, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@df.kernel
def eval_dense_solve(n: int, A: df.tensor(float), L: df.tensor(float), b: df.tensor(float), x: df.tensor(float)):
    dense_solve(n, A, L, b, x)


# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@df.kernel
def eval_dense_solve_batched(b_start: df.tensor(int),
                             A_start: df.tensor(int),
                             A_dim: df.tensor(int),
                             A: df.tensor(float),
                             L: df.tensor(float),
                             b: df.tensor(float),
                             x: df.tensor(float)):
    dense_solve_batched(b_start, A_start, A_dim, A, L, b, x)


@df.kernel
def eval_rigid_integrate(
        joint_type: df.tensor(int),
        joint_q_start: df.tensor(int),
        joint_qd_start: df.tensor(int),
        joint_q: df.tensor(float),
        joint_qd: df.tensor(float),
        joint_qdd: df.tensor(float),
        dt: float,
        # outputs
        joint_q_new: df.tensor(float),
        joint_qd_new: df.tensor(float)):

    # one thread per-articulation
    index = tid()

    type = df.load(joint_type, index)
    coord_start = df.load(joint_q_start, index)
    dof_start = df.load(joint_qd_start, index)

    jcalc_integrate(type, joint_q, joint_qd, joint_qdd,
                    coord_start, dof_start, dt, joint_q_new, joint_qd_new)


# define PyTorch autograd op to wrap simulate func
class SimulateFunc(torch.autograd.Function):
    """PyTorch autograd function representing a simulation stpe

    Note:

        This node will be inserted into the computation graph whenever
        `forward()` is called on an integrator object. It should not be called
        directly by the user.        
    """
    @staticmethod
    def forward(ctx, integrator, model, state_in, state_out, dt, update_mass_matrix, *tensors):

        # record launches
        ctx.tape = df.Tape()
        ctx.inputs = tensors
        ctx.outputs = df.to_weak_list(state_out.flatten())

        # simulate
        integrator._simulate(ctx.tape, model, state_in,
                             state_out, dt, update_mass_matrix)

        return tuple(state_out.flatten())

    @staticmethod
    def backward(ctx, *grads):

        # ensure grads are contiguous in memory
        adj_outputs = df.make_contiguous(grads)

        # register outputs with tape
        outputs = df.to_strong_list(ctx.outputs)
        for o in range(len(outputs)):
            ctx.tape.adjoints[outputs[o]] = adj_outputs[o]

        # replay launches backwards
        ctx.tape.replay()

        # find adjoint of inputs
        adj_inputs = []
        for i in ctx.inputs:

            if i in ctx.tape.adjoints:
                adj_inputs.append(ctx.tape.adjoints[i])
            else:
                adj_inputs.append(None)

        # free the tape
        ctx.tape.reset()

        # filter grads to replace empty tensors / no grad / constant params with None
        return (None, None, None, None, None, None, *df.filter_grads(adj_inputs))


class SemiImplicitIntegrator:
    """A semi-implicit integrator using symplectic Euler

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = df.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self):
        pass

    def forward(self, model: Model, state_in: State, dt: float, update_mass_matrix=True) -> State:
        """Performs a single integration step forward in time

        This method inserts a node into the PyTorch computational graph with
        references to all model and state tensors such that gradients
        can be propagrated back through the simulation step.

        Args:

            model: Simulation model
            state: Simulation state at the start the time-step
            dt: The simulation time-step (usually in seconds)

        Returns:

            The state of the system at the end of the time-step

        """

        if dflex.config.no_grad:
            state_in.knife_f.zero_()
            state_in.joint_qdd.zero_()
            
            # if no gradient required then do inplace update
            self._simulate(df.Tape(), model, state_in, state_in,
                           dt, update_mass_matrix=update_mass_matrix)
            return state_in

        else:

            # allocate new output
            state_out = model.state()

            # allocate new mass matrix if not re-using previous
            if (update_mass_matrix):
                model.alloc_mass_matrix()

            # get list of inputs and outputs for PyTorch tensor tracking
            inputs = [*state_in.flatten(), *model.flatten()]

            # run sim as a PyTorch op
            tensors = SimulateFunc.apply(
                self, model, state_in, state_out, dt, update_mass_matrix, *inputs)

            return state_out

    def _simulate(self, tape, model, state_in, state_out, dt, update_mass_matrix=True):

        with dflex.util.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                state_out.particle_f.zero_()

            if (model.link_count):
                state_out.body_ft_s = torch.zeros(
                    (model.link_count, 6), dtype=torch.float32, device=model.adapter, requires_grad=True)
                state_out.body_f_ext_s = torch.zeros(
                    (model.link_count, 6), dtype=torch.float32, device=model.adapter, requires_grad=True)

            # damped springs
            if (model.spring_count):

                tape.launch(func=eval_springs,
                            dim=model.spring_count,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.spring_indices,
                                model.spring_rest_length,
                                model.spring_stiffness,
                                model.spring_damping
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

            # triangle elastic and lift/drag forces
            if (model.tri_count and model.tri_ke > 0.0):

                tape.launch(func=eval_triangles,
                            dim=model.tri_count,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.tri_indices,
                                model.tri_poses,
                                model.tri_activations,
                                model.tri_ke,
                                model.tri_ka,
                                model.tri_kd,
                                model.tri_drag,
                                model.tri_lift
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

            # triangle/triangle contacts
            if (model.enable_tri_collisions and model.tri_count and model.tri_ke > 0.0):
                tape.launch(func=eval_triangles_contact,
                            dim=model.tri_count * model.particle_count,
                            inputs=[
                                model.particle_count,
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.tri_indices,
                                model.tri_poses,
                                model.tri_activations,
                                model.tri_ke,
                                model.tri_ka,
                                model.tri_kd,
                                model.tri_drag,
                                model.tri_lift
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

            # triangle bending
            if (model.edge_count):

                tape.launch(func=eval_bending,
                            dim=model.edge_count,
                            inputs=[state_in.particle_q, state_in.particle_qd, model.edge_indices,
                                    model.edge_rest_angle, model.edge_ke, model.edge_kd],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

            # particle ground contact
            if (model.ground and model.particle_count):

                tape.launch(func=eval_contacts,
                            dim=model.particle_count,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.contact_ke,
                                model.contact_kd,
                                model.contact_kf,
                                model.contact_mu,
                                model.contact_distance,
                                model.contact_mask
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

                with dflex.util.ScopedTimer("eval_hard_particle_contacts", False):
                    if (model.cut_edge_count):
                        # assert(not torch.isnan(f_particle).any())
                        tape.launch(
                            func=eval_hard_particle_contacts,
                            dim=model.cut_edge_count,
                            inputs=[
                                model.cut_edge_indices,
                                model.cut_edge_coords,
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.contact_ke,
                                model.contact_kd,
                                model.contact_kf,
                                model.contact_mu,
                                # TODO verify this is correct (was contact_closeness)
                                model.contact_distance
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

            # ----------------------------
            # articulations

            if (model.link_count):

                # evaluate body transforms
                tape.launch(func=eval_rigid_fk,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_joint_start,
                                model.joint_type,
                                model.joint_parent,
                                model.joint_q_start,
                                model.joint_qd_start,
                                state_in.joint_q,
                                model.joint_X_pj,
                                model.joint_X_cm,
                                model.joint_axis
                            ],
                            outputs=[state_out.body_X_sc, state_out.body_X_sm],
                            adapter=model.adapter,
                            preserve_output=True)

                # evaluate joint inertias, motion vectors, and forces
                tape.launch(func=eval_rigid_id,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_joint_start,
                                model.joint_type,
                                model.joint_parent,
                                model.joint_q_start,
                                model.joint_qd_start,
                                state_in.joint_q,
                                state_in.joint_qd,
                                model.joint_axis,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                model.body_I_m,
                                state_out.body_X_sc,
                                state_out.body_X_sm,
                                model.joint_X_pj,
                                model.gravity
                            ],
                            outputs=[
                                state_out.joint_S_s,
                                state_out.body_I_s,
                                state_out.body_v_s,
                                state_out.body_f_s,
                                state_out.body_a_s,
                            ],
                            adapter=model.adapter,
                            preserve_output=True)

                if (model.ground and model.contact_count > 0):

                    # evaluate contact forces
                    tape.launch(func=eval_rigid_contacts_art,
                                dim=model.contact_count,
                                inputs=[
                                    state_out.body_X_sc,
                                    state_out.body_v_s,
                                    model.contact_body0,
                                    model.contact_point0,
                                    model.contact_dist,
                                    model.contact_material,
                                    model.shape_materials
                                ],
                                outputs=[state_out.body_f_s],
                                adapter=model.adapter,
                                preserve_output=True)

            if (model.cut_spring_count and model.shape_count):
                sdf_based_contact = True
                if sdf_based_contact:
                    with dflex.util.ScopedTimer("eval_spring_sdf_contacts", False):
                        tape.launch(
                            func=eval_spring_sdf_contacts,
                            dim=model.cut_spring_count,
                            inputs=[
                                model.cut_spring_indices,
                                model.cut_edge_indices,
                                model.cut_edge_coords,
                                model.cut_spring_count,
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.knife_link_index,
                                state_out.body_X_sc,
                                state_out.body_v_s,
                                model.knife_params,
                                model.sdf_ke,
                                model.sdf_kd,
                                model.sdf_kf,
                                model.sdf_mu,
                                model.sdf_radius,
                            ],
                            outputs=[
                                state_out.particle_f,
                                state_out.knife_f,
                                model.cut_edge_contact_coord,
                                model.cut_edge_contact_dist,
                                model.cut_edge_contact_normal,
                            ],
                            adapter=model.adapter)
                else:
                    # collision response for a knife represented by triangular mesh
                    with dflex.util.ScopedTimer("eval_spring_tri_collision", False):
                        assert not np.isnan(
                            model.cut_edge_contact_coord.detach().cpu().numpy()).any()
                        # reset contact distances
                        model.cut_edge_contact_dist.fill_(1.0)
                        model.cut_edge_contact_coord.fill_(1.0)
                        tape.launch(
                            func=eval_spring_tri_collision,
                            dim=model.cut_spring_count * model.knife_tri_count,
                            inputs=[
                                model.cut_spring_indices,
                                model.cut_edge_indices,
                                model.cut_edge_coords,
                                model.cut_spring_count,
                                state_in.particle_q,
                                model.sdf_radius,
                                model.knife_tri_indices,
                                model.knife_tri_vertices,
                                model.knife_link_index,
                                state_out.body_X_sc,
                            ],
                            outputs=[model.cut_edge_contact_coord,
                                     model.cut_edge_contact_dist, model.cut_edge_contact_normal],
                            adapter=model.adapter)
                    # print("min contact distance:", model.cut_edge_contact_dist.min().item())
                    # count distances less than zero
                    # print("count of negative distances:", (model.cut_edge_contact_dist < 0).sum().item(), '/', model.cut_edge_contact_dist.size(0))
                    # print("contact distances:\n", model.cut_edge_contact_dist.detach().cpu().numpy())
                    # assert not np.isnan(model.cut_edge_contact_coord.detach().cpu().numpy()).any(), "cut_edge_contact_coord has NaN!"
                    # assert not np.isnan(model.cut_edge_contact_dist.detach().cpu().numpy()).any(), "cut_edge_contact_dist has NaN!"
                    # assert not np.isnan(model.cut_edge_contact_normal.detach().cpu().numpy()).any(), "cut_edge_contact_normal has NaN!"

                    with dflex.util.ScopedTimer("eval_spring_tri_contacts", False):
                        tape.launch(
                            func=eval_spring_tri_contacts,
                            dim=model.cut_spring_count,
                            inputs=[
                                model.cut_spring_indices,
                                model.cut_edge_indices,
                                model.cut_edge_coords,
                                model.cut_spring_count,
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.knife_link_index,
                                state_out.body_X_sc,
                                state_out.body_v_s,
                                model.cut_edge_contact_coord,
                                model.cut_edge_contact_dist,
                                model.cut_edge_contact_normal,
                                model.sdf_ke,
                                model.sdf_kd,
                                model.sdf_kf,
                                model.sdf_mu,
                                model.sdf_radius,
                            ],
                            outputs=[
                                state_out.particle_f,
                                state_out.knife_f,
                            ],
                            adapter=model.adapter)

                        assert not np.isnan(state_out.particle_f.detach(
                        ).cpu().numpy()).any(), "particle_f has NaN!"

                # assert(not torch.isnan(state_out.particle_f).any())
                # assert(not torch.isnan(state_out.knife_f).any())
                # assert(not torch.isnan(model.cut_spring_contact_coords).any())

                # assert(not torch.isnan(state_in.q).any())
                # assert(not torch.isnan(state_in.u).any())
                # assert(not torch.isnan(model.cut_edge_coords).any())
                # assert(not torch.isnan(model.cut_spring_rest_length).any())
                # assert(not torch.isnan(state_in.cut_spring_ke).any())
                # assert(not torch.isnan(state_in.cut_spring_kd).any())
                # assert(not torch.isnan(model.cut_spring_normal).any())
                # assert(not torch.isnan(model.cut_spring_contact_stiffness).any())
                # assert(not torch.isnan(model.cut_spring_softness).any())
                # assert(not torch.isnan(state_out.knife_f).any())
                # assert(not torch.isnan(state_out.particle_f).any())
                # assert(not torch.isnan(state_out.cut_spring_ke).any())
                # assert(not torch.isnan(state_out.cut_spring_kd).any())
                with dflex.util.ScopedTimer("eval_cut_springs", False):
                    tape.launch(
                        func=eval_cut_springs,
                        dim=model.cut_spring_count,
                        inputs=[
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.cut_edge_indices,
                            model.cut_edge_coords,
                            model.cut_spring_indices,
                            model.cut_spring_rest_length,
                            state_in.cut_spring_ke,
                            state_in.cut_spring_kd,
                            model.cut_spring_normal,
                            model.sdf_ke,
                            model.cut_spring_softness,
                            state_out.knife_f,
                        ],
                        outputs=[
                            state_out.particle_f,
                            state_out.cut_spring_ke,                 # state_out.cut_spring_kd,
                        ],
                        adapter=model.adapter)

            # tetrahedral FEM
            if (model.tet_count):

                tape.launch(func=eval_tetrahedra,
                            dim=model.tet_count,
                            inputs=[
                                state_in.particle_q,
                                state_in.particle_qd,
                                model.tet_indices,
                                model.tet_poses,
                                model.tet_activations,
                                model.tet_mu,
                                model.tet_lambda,
                                model.tet_damping
                            ],
                            outputs=[state_out.particle_f],
                            adapter=model.adapter)

                # particle shape contact
                if (model.particle_count):

                    # tape.launch(func=eval_soft_contacts,
                    #             dim=model.particle_count*model.shape_count,
                    #             inputs=[state_in.particle_q, state_in.particle_qd, model.contact_ke, model.contact_kd, model.contact_kf, model.contact_mu],
                    #             outputs=[state_out.particle_f],
                    #             adapter=model.adapter)

                    tape.launch(
                        func=eval_soft_contacts,
                        dim=model.particle_count * model.shape_count,
                        inputs=[
                            model.particle_count,
                            state_in.particle_q,
                            state_in.particle_qd,
                            state_in.body_X_sc,
                            state_in.body_v_s,
                            model.shape_transform,
                            model.shape_body,
                            model.shape_geo_type,
                            torch.Tensor(),
                            model.shape_geo_scale,
                            model.shape_materials,
                            model.contact_ke,
                            model.contact_kd,
                            model.contact_kf,
                            model.contact_mu
                        ],
                        # outputs
                        outputs=[state_out.particle_f, state_out.body_f_s],
                        adapter=model.adapter)

                # evaluate muscle actuation
                # tape.launch(func=eval_muscles,
                #             dim=model.muscle_count,
                #             inputs=[
                #                 state_out.body_X_sc,
                #                 state_out.body_v_s,
                #                 model.muscle_start,
                #                 model.muscle_params,
                #                 model.muscle_links,
                #                 model.muscle_points,
                #                 model.muscle_activation
                #             ],
                #             outputs=[state_out.body_f_s],
                #             adapter=model.adapter,
                #             preserve_output=True)

                # # evaluate joint torques
                # tape.launch(func=eval_rigid_tau,
                #             dim=model.articulation_count,
                #             inputs=[
                #                 model.articulation_joint_start,
                #                 model.joint_type,
                #                 model.joint_parent,
                #                 model.joint_q_start,
                #                 model.joint_qd_start,
                #                 state_in.joint_q,
                #                 state_in.joint_qd,
                #                 state_in.joint_act,
                #                 model.joint_target,
                #                 model.joint_target_ke,
                #                 model.joint_target_kd,
                #                 model.joint_limit_lower,
                #                 model.joint_limit_upper,
                #                 model.joint_limit_ke,
                #                 model.joint_limit_kd,
                #                 model.joint_axis,
                #                 state_out.joint_S_s,
                #                 state_out.body_f_s
                #             ],
                #             outputs=[state_out.body_ft_s, state_out.joint_tau],
                #             adapter=model.adapter,
                #             preserve_output=True)

                # if (update_mass_matrix):

                #     # build J
                #     tape.launch(
                #         func=eval_rigid_jacobian,
                #         dim=model.articulation_count,
                #         inputs=[
                #             # inputs
                #             model.articulation_joint_start,
                #             model.articulation_J_start,
                #             model.joint_parent,
                #             model.joint_qd_start,
                #             state_out.joint_S_s
                #         ],
                #         outputs=[model.J],
                #         adapter=model.adapter,
                #         preserve_output=True)

                #     # build M
                #     tape.launch(
                #         func=eval_rigid_mass,
                #         dim=model.articulation_count,
                #         inputs=[
                #             # inputs
                #             model.articulation_joint_start,
                #             model.articulation_M_start,
                #             state_out.body_I_s
                #         ],
                #         outputs=[model.M],
                #         adapter=model.adapter,
                #         preserve_output=True)

                #     # form P = M*J
                #     df.matmul_batched(
                #         tape,
                #         model.articulation_count,
                #         model.articulation_M_rows,
                #         model.articulation_J_cols,
                #         model.articulation_J_rows,
                #         0,
                #         0,
                #         model.articulation_M_start,
                #         model.articulation_J_start,
                #         # P start is the same as J start since it has the same dims as J
                #         model.articulation_J_start,
                #         model.M,
                #         model.J,
                #         model.P,
                #         adapter=model.adapter)

                #     #    model.joint_dof_count,
                #     #     model.joint_dof_count,
                #     #     model.link_count*6,

                #     # form H = J^T*P
                #     df.matmul_batched(
                #         tape,
                #         model.articulation_count,
                #         model.articulation_J_cols,
                #         model.articulation_J_cols,
                #         model.articulation_J_rows,         # P rows is the same as J rows
                #         1,
                #         0,
                #         model.articulation_J_start,
                #         # P start is the same as J start since it has the same dims as J
                #         model.articulation_J_start,
                #         model.articulation_H_start,
                #         model.J,
                #         model.P,
                #         model.H,
                #         adapter=model.adapter)

                #     # compute decomposition
                #     tape.launch(func=eval_dense_cholesky_batched,
                #                 dim=model.articulation_count,
                #                 inputs=[
                #                     model.articulation_H_start, model.articulation_H_rows, model.H, model.joint_armature],
                #                 outputs=[model.L],
                #                 adapter=model.adapter,
                #                 skip_check_grad=True)

                # # solve for qdd
                # tape.launch(func=eval_dense_solve_batched,
                #             dim=model.articulation_count,
                #             inputs=[model.articulation_dof_start, model.articulation_H_start,
                #                     model.articulation_H_rows, model.H, model.L, state_out.joint_tau],
                #             outputs=[state_out.joint_qdd],
                #             adapter=model.adapter,
                #             skip_check_grad=True)

                # set knife acceleration to zero since it follows a prescribed position/velocity trajectory
                # state_out.joint_qdd[:6].zero_()
                # state_out.joint_qdd.zero_()

                # integrate joint dofs -> joint coords
                tape.launch(func=eval_rigid_integrate,
                            dim=model.link_count,
                            inputs=[model.joint_type, model.joint_q_start, model.joint_qd_start,
                                    state_in.joint_q, state_in.joint_qd, state_out.joint_qdd, dt],
                            outputs=[state_out.joint_q, state_out.joint_qd],
                            adapter=model.adapter)

                # assert not np.isnan(state_out.joint_q.detach().cpu().numpy()).any() and not np.isnan(
                #     state_out.joint_qd.detach().cpu().numpy()).any(), "Joint q or qd is NaN!"

            # ----------------------------
            # integrate particles

            if (model.particle_count):
                tape.launch(func=integrate_particles,
                            dim=model.particle_count,
                            inputs=[state_in.particle_q, state_in.particle_qd,
                                    state_out.particle_f, model.particle_inv_mass, model.gravity, dt],
                            outputs=[state_out.particle_q,
                                     state_out.particle_qd],
                            adapter=model.adapter)

            return state_out

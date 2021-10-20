# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# fmt: off
from abc import abstractmethod
import os
import sys
from cutting.utils import as_tensor
import numpy as np
import torch
import json

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
# fmt: on


class Motion:
    """
    Implements the motion of the knife, i.e. it determines linear + angular position and velocity at each time step.
    """
    @abstractmethod
    def update_state(self, state, time, dt):
        pass


class FreeFloatingKnifeMotion(Motion):
    """
    Free-floating knife motion where the knife orientation is defined by a quaternion, linear position and angula position and velocities are defined by 3D vectors.
    """
    @abstractmethod
    def linear_position(self, time, dt):
        pass

    @abstractmethod
    def angular_position(self, time, dt):
        pass

    @abstractmethod
    def linear_velocity(self, time, dt):
        pass

    @abstractmethod
    def angular_velocity(self, time, dt):
        pass

    def update_state(self, state, time, dt):
        if time < dt * 2:
            state.joint_q[0:3] = self.linear_position(time, dt)
            state.joint_q[3:7] = self.angular_position(time, dt)

        state.joint_qd[0:3] = self.linear_velocity(time, dt)
        state.joint_qd[3:6] = self.angular_velocity(time, dt)


class ConstantLinearVelocityMotion(FreeFloatingKnifeMotion):
    """
    Constant linear knife motion starting from an initial position and orientation.
    """

    def __init__(self, initial_pos, linear_velocity, initial_rot=np.array([0., 0., 0., 1.]), device='cuda'):
        self.initial_pos = as_tensor(initial_pos, device=device)
        self.lin_vel = as_tensor(
            linear_velocity, device=self.initial_pos.device)
        self.initial_rot = as_tensor(
            initial_rot, device=self.initial_pos.device)
        self.device = device

    def linear_position(self, time, dt):
        return self.initial_pos + time * self.lin_vel

    def angular_position(self, time, dt):
        return self.initial_rot

    def linear_velocity(self, time, dt):
        return self.lin_vel

    def angular_velocity(self, time, dt):
        return torch.zeros(3, device=self.initial_pos.device)


class ForwardFacingMotion(FreeFloatingKnifeMotion):
    """
    Knife motion that is defined by a linear motion trajectory and the blade is aligned with the tangent of the linear motion.
    """

    def __init__(self, linear_motion: Motion):
        self.linear_motion = linear_motion
        self.device = linear_motion.device
        self.__last_time = -1.0
        self.__last_tangent = None

    def linear_position(self, t, dt):
        return self.linear_motion.linear_position(t)

    def linear_velocity(self, t, dt):
        if self.__last_time == t:
            return self.__last_tangent
        self.__last_time = t
        self.__last_tangent = self.linear_motion.linear_velocity(t)
        return self.__last_tangent

    def angular_position(self, t, dt):
        import dflex as df
        # TODO implement in pytorch
        up = np.array((1., 0., 0.))
        tangent = self.linear_velocity(t, dt).detach().cpu().numpy()
        tangent /= np.linalg.norm(tangent)
        left = np.cross(tangent, up)
        if np.linalg.norm(left) <= 1e-8:
            return torch.tensor(df.quat_rpy(-np.pi / 2, np.pi / 2, 0), device=self.device)
        left /= np.linalg.norm(left)
        angle = np.arccos(np.dot(up, tangent))
        quat = df.quat_multiply(df.quat_from_axis_angle(
            left, -angle), df.quat_rpy(-np.pi / 2, np.pi / 2, -0.0))
        # quat = df.quat_from_axis_angle(left, angle)
        return torch.tensor(quat, device=self.device)

    def angular_velocity(self, t: torch.Tensor, dt: float):
        import dflex as df
        # TODO implement in pytorch
        t1 = t - dt / 2
        lin_vel_1 = self.linear_velocity(t1, dt).detach().cpu().numpy()
        lin_vel_1 /= np.linalg.norm(lin_vel_1)
        q1 = self.angular_position(t1, dt)

        t2 = t + dt / 2
        lin_vel_2 = self.linear_velocity(t2, dt).detach().cpu().numpy()
        lin_vel_2 /= np.linalg.norm(lin_vel_2)
        q2 = self.angular_position(t2, dt)

        qd = df.quat_multiply(((q2 - q1) / dt * 2).detach().cpu().numpy(),
                              df.quat_inverse(q1.detach().cpu().numpy()))
        return torch.tensor(qd[:3], device=self.device)


class SplineForwardFacingMotion(ForwardFacingMotion):
    """
    Knife motion that is defined by a linear motion trajectory and the blade is aligned with the tangent of the linear motion.
    The linear motion is defined by a spline, e.g. a cubic spline from the torchcubicspline package.
    """

    def __init__(self, spline, device):
        self.device = device
        self.spline = spline
        # self.__last_time = -1.0
        self.__last_tangent = None
        super(SplineForwardFacingMotion, self).__init__(linear_motion=self)

    def linear_position(self, t, dt):
        return self.spline.evaluate(as_tensor(t, device=self.device))

    def linear_velocity(self, t, dt):
        # if self.__last_time == t:
        #     return self.__last_tangent
        # self.__last_time = t
        self.__last_tangent = self.spline.derivative(as_tensor(t, device=self.device))
        # print("vel", self.__last_tangent)
        # print("time", t)
        return self.__last_tangent


class SlicingMotion(FreeFloatingKnifeMotion):
    """
    Slicing motion where the knife presses vertically downwards (negative y), while moving laterally (along z) in a sine wave.
    """

    def __init__(self,
                 initial_pos,
                 slicing_frequency,
                 slicing_amplitudes,
                 pressing_velocities,
                 slicing_times,
                 slicing_kernel_width,
                 initial_rot=np.array([0., 0., 0., 1.]),
                 device='cuda'):
        assert len(slicing_times) == len(slicing_amplitudes) == len(
            pressing_velocities), "Number of slicing times, amplitudes and velocities must be equal"
        self.initial_pos = as_tensor(initial_pos, device=device)
        self.slicing_frequency = as_tensor(
            slicing_frequency, device=self.initial_pos.device)
        self.slicing_amplitudes = as_tensor(
            slicing_amplitudes, device=self.initial_pos.device)
        self.pressing_velocities = as_tensor(
            pressing_velocities, device=self.initial_pos.device)
        self.slicing_times = as_tensor(
            slicing_times, device=self.initial_pos.device)
        self.slicing_kernel_width = slicing_kernel_width
        self.initial_rot = as_tensor(
            initial_rot, device=self.initial_pos.device)
        self.device = device

    def get_weighting(self, time):
        # RBF kernel weighting
        return torch.exp(-(time - self.slicing_times)**2. / self.slicing_kernel_width)

    def linear_position(self, time, dt):
        # just an approximation which is good enough for the first few time steps
        return self.initial_pos + time * self.linear_velocity(time, dt)

    def angular_position(self, time, dt):
        return self.initial_rot

    def linear_velocity(self, time, dt):
        weighting = self.get_weighting(time)
        amp = torch.dot(weighting, self.slicing_amplitudes)
        v = torch.zeros(3, device=self.device)
        v[1] = torch.dot(weighting, self.pressing_velocities)
        v[2] = torch.sin(time * self.slicing_frequency) * amp
        return v

    def angular_velocity(self, time, dt):
        return torch.zeros(3, device=self.device)

    def plot(self, sim_duration: float):
        """
        Plot keypoint weighting and amplitude/velocity profile of the slicing motion.
        """
        import matplotlib.pyplot as plt
        num_waypoints = len(self.slicing_times)
        waypoints = np.linspace(0.0, sim_duration, num_waypoints)
        ts = np.linspace(0.0, sim_duration, 500)
        total = np.zeros(500)

        fig = plt.figure()
        plt.subplot(311)
        plt.ylim([0.0, 1.0])
        for w in waypoints:
            plt.axvline(w, color='k', linestyle='--')
        for w in waypoints:
            kernel = np.exp(-(ts - w)**2. / self.slicing_kernel_width)
            total += kernel
            plt.plot(ts, kernel)
            plt.fill_between(ts, kernel, 0., alpha=0.3)
        plt.xticks(waypoints)
        plt.gca().set_xticklabels(np.arange(num_waypoints))
        plt.title("Keypoints")
        plt.ylabel("Weight")

        plt.subplot(312)
        plt.title("Amplitude")
        plt.xlabel("Time [s]")
        plt.ylabel("$\mathbf{a}(t)$")
        trajectory = self.slicing_amplitudes.detach().cpu().numpy()
        if len(trajectory) >= num_waypoints:
            interpolated = []
            for t in ts:
                weighting = np.exp(-(t - waypoints)**2. /
                                   self.slicing_kernel_width)
                interpolated.append(
                    np.dot(weighting, trajectory[:num_waypoints]))
            plt.plot(ts, interpolated, linewidth=3,
                     color="gray", linestyle="--", zorder=1)
            for i in range(len(trajectory)):
                plt.scatter([waypoints[i]], [trajectory[i]],
                            color="white", s=100, zorder=2)
            for i in range(len(trajectory)):
                plt.scatter([waypoints[i]], [trajectory[i]], zorder=3)
        else:
            plt.plot(ts, total)
        plt.grid()

        plt.subplot(313)
        plt.title("Pressing Velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("$\mathbf{a}(t)$")
        trajectory = self.pressing_velocities.detach().cpu().numpy()
        if len(trajectory) >= num_waypoints:
            interpolated = []
            for t in ts:
                weighting = np.exp(-(t - waypoints)**2. /
                                   self.slicing_kernel_width)
                interpolated.append(
                    np.dot(weighting, trajectory[:num_waypoints]))
            plt.plot(ts, interpolated, linewidth=3,
                     color="gray", linestyle="--", zorder=1)
            for i in range(len(trajectory)):
                plt.scatter([waypoints[i]], [trajectory[i]],
                            color="white", s=100, zorder=2)
            for i in range(len(trajectory)):
                plt.scatter([waypoints[i]], [trajectory[i]], zorder=3)
        else:
            plt.plot(ts, total)
        plt.grid()

        plt.tight_layout()
        return fig


class RobotMotion(Motion):
    """
    Defines a knife motion for a fixed-base mechanism (e.g. robot arm) where the knife is attached to link.
    The motion is applied as joint velocities/positions to the robot.
    """

    def __init__(self, start_q_index: int, dof: int, dof_qd: int = None):
        self.start_q_index = start_q_index
        self.dof = dof
        self.dof_qd = dof_qd if dof_qd is not None else dof

    @abstractmethod
    def joint_q(self, time, dt):
        pass

    @abstractmethod
    def joint_qd(self, time, dt):
        pass

    def update_state(self, state, time, dt):
        if time < dt * 2:
            state.joint_q[self.start_q_index:self.start_q_index +
                          self.dof] = self.joint_q(time, dt)
        state.joint_qd[self.start_q_index:self.start_q_index +
                       self.dof_qd] = self.joint_qd(time, dt)


class RobotMotionFromJSON(RobotMotion):
    """
    Load a predefined robot motion (joint positions and velocities) from a JSON file.
    Joint positions and velocities are assumed to be at keys "q" and "qd", time is assumed to be at key "time".
    """

    def __init__(self, filename: str, start_q_index: int = 0, time_scaling: float = 1.0, device='cuda'):
        with open(filename, "r") as f:
            self.data = json.load(f)
        self.q = as_tensor(self.data["q"], device=device)
        self.qd = as_tensor(self.data["qd"], device=device)
        self.time = as_tensor(self.data["time"], device=device)
        # we assume that time is recorded at equidistant time steps
        self.dt = self.time[1] - self.time[0]
        self.time_scaling = time_scaling
        super().__init__(start_q_index, self.q.shape[1], self.qd.shape[1])

    def joint_q(self, time, dt):
        t = time * self.time_scaling
        if t <= self.time[0]:
            return self.q[0]
        i = int((t - self.time[0].item()) / self.dt.item())
        if t >= self.time[-1] or i >= len(self.time) - 1:
            return self.q[-1]
        frac = (t - self.time[i].item()) / self.dt.item()
        # linear interpolation
        return (1 - frac) * self.q[i] + frac * self.q[i + 1]

    def joint_qd(self, time, dt):
        t = time * self.time_scaling
        if t <= self.time[0]:
            return self.qd[0]
        i = int((t - self.time[0].item()) / self.dt.item())
        if t >= self.time[-1] or i >= len(self.time) - 1:
            return torch.zeros_like(self.qd[-1])
        frac = (t - self.time[i].item()) / self.dt.item()
        # linear interpolation
        return ((1 - frac) * self.qd[i] + frac * self.qd[i + 1]) * self.time_scaling

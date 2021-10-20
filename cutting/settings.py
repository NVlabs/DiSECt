# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import json
import os
from munch import *
from copy import deepcopy
from collections import MutableMapping, namedtuple, defaultdict
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from glob import glob


class Parameter:
    """
    Simulation parameter which has a name, a value, and a range.
    By default, the parameter limits are enforced by a sigmoid function.
    This class maps the parameter to its tensor which can be optimized.
    """

    def __init__(self, name, value, low=0., high=1., individual=True, fixed=False, apply_sigmoid=True):
        self.name = name
        # whether this parameter is optimized
        self.fixed = fixed
        self.low = low
        self.high = high
        # whether this parameter is replicated across the tensor, or can take up different numbers
        self.individual = individual
        self.tensor = None
        self.apply_sigmoid = apply_sigmoid
        self.shape = None

        self.value = value

    def sample(self):
        if np.isinf(self.low) or np.isinf(self.high):
            return np.random.randn() * self.value + self.value
        else:
            return np.random.uniform(self.low, self.high)

    # custom assignment operator that converts the float value to sigmoid space
    def set_value(self, value):
        assert type(value) is float or type(
            value) is int, f"{self.name} is not a float"
        assert (
            value >= self.low), f"{self.name} is out of bounds ({value} !>= {self.low})"
        assert (
            value <= self.high), f"{self.name} is out of bounds ({value} !<= {self.high})"
        if self.apply_sigmoid:
            value = Parameter.inverse_sigmoid(
                (float(value) - self.low) / self.range)
        self.value = value

    @property
    def normalized_value(self):
        return (self.actual_value - self.low) / self.range

    @property
    def range(self):
        return self.high - self.low

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def inverse_sigmoid(y):
        return np.log(y / (1. - y + 1e-6))

    @property
    def actual_value(self):
        if self.apply_sigmoid:
            return Parameter.sigmoid(self.value) * self.range + self.low
        else:
            return self.value

    @property
    def actual_tensor_value(self):
        if self.tensor is None:
            return None
        val = self.tensor.detach().cpu().numpy()
        if self.apply_sigmoid:
            return Parameter.sigmoid(val) * self.range + self.low
        else:
            return val

    def initial_value(self, settings):
        return settings[self.name]
        # if np.isinf(self.low) or np.isinf(self.high):
        #     return value
        # return Parameter.inverse_sigmoid((value - self.low) / self.range)

    def apply_bounds(self, tensor):
        if np.isinf(self.low) or np.isinf(self.high):
            return tensor
        return torch.sigmoid(tensor) * self.range + self.low

    def create_tensor(self, src_tensor, adapter):
        self.shape = None
        dtype = torch.float32
        if src_tensor is not None:
            self.shape = src_tensor.shape
        if src_tensor is None or self.fixed:
            assert np.any(np.array(
                self.value) >= self.low), f"{self.name} is out of bounds ({self.value} !>= {self.low})"
            assert np.any(np.array(
                self.value) <= self.high), f"{self.name} is out of bounds ({self.value} !<= {self.high})"

            if self.apply_sigmoid:
                value = Parameter.inverse_sigmoid(
                    (np.array(self.value) - self.low) / self.range)
            else:
                value = self.value
            self.tensor = torch.tensor(
                value, device=adapter, requires_grad=True)
            if self.shape is None:
                self.shape = self.tensor.shape
            return self.tensor
        else:
            dtype = src_tensor.dtype
        assert (not torch.isnan(src_tensor).any()), f"{self.name} is NaN"

        if self.individual:
            values = src_tensor.detach().cpu().numpy()
            assert (np.min(
                values) >= self.low), f"{self.name} is out of bounds ({np.min(values)} !>= {self.low})"
            assert (np.max(
                values) <= self.high), f"{self.name} is out of bounds ({np.max(values)} !<= {self.high})"
            if self.apply_sigmoid:
                # convert to sigmoid space
                values = Parameter.inverse_sigmoid(
                    (values - self.low) / self.range)
            self.tensor = torch.tensor(
                values, dtype=dtype, device=adapter, requires_grad=True)
        else:
            # create scalar tensor from the mean of the input
            self.value = torch.mean(src_tensor.detach()).item()
            assert (
                self.value >= self.low), f"{self.name} is out of bounds ({self.value} !>= {self.low})"
            assert (
                self.value <= self.high), f"{self.name} is out of bounds ({self.value} !<= {self.high})"
            if self.apply_sigmoid:
                value = Parameter.inverse_sigmoid(
                    (self.value - self.low) / self.range)
            else:
                value = self.value
            self.tensor = torch.tensor(
                value, dtype=dtype, device=adapter, requires_grad=True)
        assert (not torch.isnan(self.tensor).any()), f"{self.name} is NaN"
        return self.tensor

    def assignable_tensor(self):
        if self.tensor is None:
            raise Exception(
                f"Tensor for parameter {self.name} has not been set.")
        if self.apply_sigmoid:
            self.op = self.apply_bounds(self.tensor)
        else:
            self.op = self.tensor
        dim = "x".join(map(str, self.shape)) if self.tensor.ndim > 0 else "1"
        if self.op.ndim > 0:
            print(
                f'Assigning {self.name} = [{torch.min(self.op).item()} .. {torch.max(self.op).item()}]\t (initial value: [{np.min(self.value)} .. {np.max(self.value)}], actual tensor value: [{np.min(self.actual_tensor_value)} .. {np.max(self.actual_tensor_value)}]) with dimension {dim}.'
            )
        else:
            print(
                f'Assigning {self.name} = {self.op.item()}\t (initial value: {self.value}, actual tensor value: {self.actual_tensor_value}) with dimension {dim}.'
            )
        if not self.individual and self.shape is not None:
            if self.tensor.ndim > 0:
                raise Exception(
                    f"Tensor for parameter {self.name} is marked as repeated (not individual) but not a scalar.")
            self.expanded_tensor = self.op.expand(self.shape).contiguous()
            return self.expanded_tensor
        return self.op

    def print_tensor(self):
        if self.tensor is None:
            return
        if self.apply_sigmoid:
            op = self.apply_bounds(self.tensor)
        else:
            op = self.tensor
        if self.tensor.ndim == 0:
            print(f"\t{self.name} = {op.item()}")
            return
        with torch.no_grad():
            tmin = torch.min(op)
            tmax = torch.max(op)
            print(f"\t{self.name} = {tmin.item()} ... {tmax.item()}")

    def __str__(self):
        return f"{self.name} [{self.low}  {self.high}]"

    def __repr__(self):
        return f"{self.name} = {self.value} [{self.low}  {self.high}]"


default_parameters = {
    "velocity_y": Parameter("velocity_y", -0.05, -0.07, -0.03),
    "initial_y": Parameter("initial_y", 0.08, 0.065, 0.09),
    "lateral_velocity": Parameter("lateral_velocity", 0.0, -0.07, 0.07, True, False, apply_sigmoid=False),
    # "cut_spring_kd": Parameter("cut_spring_kd", 0.1, 0.01, 0.15),
    "cut_spring_ke": Parameter("cut_spring_ke", 500, 100, 8000),
    "cut_spring_softness": Parameter("cut_spring_softness", 500, 10, 5000),
    "sdf_radius": Parameter("sdf_radius", 0.5e-3, 0.1e-3, 1.2e-3),
    "sdf_ke": Parameter("sdf_ke", 4000, 500., 8000, individual=True),
    "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
    "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
    "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
    "mu": Parameter("mu", 0.69e6, 0.1e6, 1.5e6, individual=True),
    "lambda": Parameter("lambda", 6.2e6, 0.1e5, 8e6, individual=True),
    "density": Parameter("density", 700, 500., 1000., individual=True),
    "damping": Parameter("damping", 1., 0.1, 10., individual=True),
}

default_settings = munchify({
    "seed": 42,
    "integrator": "explicit",
    "sim_duration": 0.9,
    # render at every n-th step (to generate USD file)
    "sim_substeps": 500,
    "sim_dt": 1e-5,
    "ground": True,
    "ground_ke": 100.,
    "ground_kd": 0.1,
    "ground_kf": 0.2,
    "ground_mu": 0.6,
    "ground_radius": 1e-3,
    "generator": "ansys",
    "generators": {
        "ansys": {
            "filename": "assets/sphere.rst"
        },
        "meshio": {
            "filename": "assets/apple1.obj_.msh"
        },
        "grid": {
            "dim_x": 21,
            "dim_y": 8,
            "dim_z": 8,
            "cell_x": 0.006,
            "cell_y": 0.006,
            "cell_z": 0.006,
        }
    },                                                                                                                  # apple
    "mu": 689655,
    "lambda": 6206896,
    "young": 3e6,
    "poisson": 0.17,
    "density": 787.,
    "minimum_particle_mass": 0.0,
    "damping": 10.,
    "relaxation": 1.0,
    "geometry": {
        "position": (0., 0.0005, 0.),
        # axis (xyz), angle
        "rotation": (0., 0., 1., 0.),
        "scale": 1.
    },
    "cut_spring_ke": 5e2,
    "cut_spring_kd": 0.1,
    "cut_spring_softness": 500.,
    "cut_spring_rest_length": 0.,
    "surface_cut_spring_ke": 8e2,
    "surface_cut_spring_kd": 0.1,
    "surface_cut_spring_softness": 20.,
    "surface_cut_spring_rest_length": 0.,
    "sdf_ke": 4000,
    "sdf_kd": 1.,
    "sdf_kf": 0.01,
    "sdf_mu": 0.5,
    "surface_sdf_ke": 4000,
    "surface_sdf_kd": 1.,
    "surface_sdf_kf": 0.01,
    "surface_sdf_mu": 0.5,
    "sdf_radius": 0.0005,
    # "knife_sdf": {
    #     "spine_dim": 0.002, "edge_dim": 0.0008, "spine_height": 0.04, "tip_height": 0.0004, "radius": 0., "depth": 0.15
    # },
    "static_vertices": {
        "include": [{
            "min": [-100., -100., -100.], "max": [100., 1e-3, 100.]
        }], "exclude": [{
            "min": [-0.01, -100., -100.], "max": [0.01, 100., 100.]
        }]
    },
    "velocity_y": -0.05,
    "initial_y": 0.08,
    "knife_motion": {
        "position": [0.0, 0.08, 0.0],
        "rotation": [0.0, 0.0, 1.0, 0.0],
        "velocity": [0.0, -0.05, 0.0],
        "omega": [0.0, 0.0, 0.0],
    },
    "knife_type": "ybj",
    "cutting": {
        "active": True,
        "mode": "fem",   # can be one of ("rigid", "hybrid", "fem")
    }
})


def convert_lame(young, poisson):
    return (young / (2 * (1.0 + poisson)), (young * poisson) / ((1.0 + poisson) * (1.0 - 2 * poisson)))


def load_settings(filename):
    """
    Loads settings from a JSON file.
    """
    if filename is None or len(filename) == 0:
        return default_settings

    def rec_merge(d1, d2):
        '''
        Update two dicts of dicts recursively, 
        if either mapping has leaves that are non-dicts, 
        the second's leaf overwrites the first's.
        '''
        # source: https://stackoverflow.com/a/24088493
        for k, v in d1.items():
            if k in d2:
                if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                    d2[k] = rec_merge(v, d2[k])
        d3 = d1.copy()
        d3.update(d2)
        return d3

    with open(filename, "r") as f:
        settings = json.load(f)
        settings = munchify(rec_merge(default_settings, settings))
        k_mu, k_lambda = convert_lame(
            young=settings.young, poisson=settings.poisson)
        settings["mu"] = k_mu
        settings["lambda"] = k_lambda
    with open(filename, "w") as f:
        f.write(json.dumps(settings, indent=4))
    return settings


def save_settings(settings, filename, silence=False):
    with open(filename, "w") as f:
        f.write(json.dumps(settings, indent=4))
        if not silence:
            print(f"Saved settings at {filename}.")


def check_configs(folder, dataset_name=None):
    files = list(glob(f'{folder}/*.json'))
    settings_0 = json.load(open(files[0], 'r'))
    diffs = defaultdict(list)
    other_diffs = set()
    for i in range(1, len(files)):
        config = json.load(open(files[i], 'r'))
        for key, v0 in settings_0.items():
            if config[key] != v0:
                if type(v0) != float:
                    other_diffs.add(key)
                else:
                    diffs[key].append(config[key])
    if dataset_name is not None:
        dataset = datasets[dataset_name]
    else:
        dataset = None
    print("Differences:")
    for name in other_diffs:
        print(f'\t{name}')
    success = True
    for key, ds in diffs.items():
        ds.append(settings_0[key])
        ds = np.array(ds)
        print(f'\t{key}:\t', np.min(ds), '...', np.max(ds))
        if dataset is not None and key not in dataset.params:
            print(
                f'\t\tError: Not found in parameters of dataset "{dataset_name}"!')
            success = False

    if dataset is not None:
        for p in dataset.params:
            if p not in diffs.keys():
                print(
                    f'Error: Dataset parameter {p} is not modulated in the config files from {folder}!')
                success = False

    return success


def generate_configs(num, params, base_config_file="", surface_equal_interior=False, prefix="gen", seed=None):
    """
    Generate configuration files for the trajectory rollouts used by BayesSim.
    """
    try:
        os.mkdir(f"config/{prefix}")
    except:
        pass
    if seed is not None:
        np.random.seed(seed)
    base_config = load_settings(base_config_file)
    dual_params = set(["sdf_ke", "sdf_kd", "sdf_kf", "sdf_mu",
                      "cut_spring_ke", "cut_spring_softness"])
    for i in tqdm(range(num)):
        settings = deepcopy(base_config)
        if surface_equal_interior:
            for param in [p for p in params if not p.startswith("surface_")]:
                settings[param] = default_parameters[param].sample()
                if param in dual_params:
                    settings[f"surface_{param}"] = settings[param]
        else:
            for param in params:
                if param.startswith("surface_"):
                    settings[param] = default_parameters[param[len(
                        "surface_"):]].sample()
                else:
                    settings[param] = default_parameters[param].sample()

        settings["knife_motion"]["position"][1] = settings["initial_y"]
        settings["knife_motion"]["velocity"][1] = settings["velocity_y"]
        settings["timestamp"] = str(datetime.now())
        save_settings(
            settings, f"config/{prefix}/{prefix}_{i:03d}.json", silence=True)
    print(f"Saved {num} configurations with prefix {prefix}.")


Dataset = namedtuple('Dataset', 'params config_file surface_equal_interior')
datasets = {
    '2d_sdf':
    Dataset(params=['sdf_ke', 'surface_sdf_ke'],
            config_file="config/default_cut_settings.json", surface_equal_interior=True),
    '4d_springs':
    Dataset(params=[
        'cut_spring_ke',
        'surface_cut_spring_ke',
        'cut_spring_softness',
        'surface_cut_spring_softness',
    ],
        config_file="config/default_cut_settings.json",
        surface_equal_interior=False),
    'cutting':
    Dataset(params=[
        'sdf_ke',
        'surface_sdf_ke',
        'sdf_kd',
        'surface_sdf_kd',
        'sdf_kf',
        'surface_sdf_kf',
        'sdf_mu',
        'surface_sdf_mu',
        'cut_spring_ke',
        'surface_cut_spring_ke',
        'cut_spring_softness',
        'surface_cut_spring_softness',
        'velocity_y',
        'initial_y'
    ],
        config_file="config/default_cut_settings.json",
        surface_equal_interior=False),
    'ansys':
    Dataset(
        params=[
            'sdf_ke',
            'sdf_kd',
            'sdf_kf',
            'sdf_mu',
            'cut_spring_ke',
            'cut_spring_softness',
            'initial_y'
        ],
        config_file="config/ansys_sphere_apple.json",
        surface_equal_interior=False),
    'real':
    Dataset(params=[
        'sdf_ke',
        'surface_sdf_ke',
        'sdf_kd',
        'surface_sdf_kd',
        'sdf_kf',
        'surface_sdf_kf',
        'sdf_mu',
        'surface_sdf_mu',
        'cut_spring_ke',
        'surface_cut_spring_ke',
        'cut_spring_softness',
        'surface_cut_spring_softness',
        'initial_y',
        'velocity_y'
    ],
        config_file="config/ybj_apple.json",
        surface_equal_interior=False),
    'actual_2d':
    Dataset(params=[
        'sdf_ke',
        'cut_spring_ke',
    ], config_file="config/default_cut_settings.json", surface_equal_interior=True),
    'actual_5d':
    Dataset(params=[
        'sdf_ke',
        'sdf_kf',
        'cut_spring_ke',
        'cut_spring_softness',
    ], config_file="config/default_cut_settings.json", surface_equal_interior=True)
}


def get_dataset_parameters(dataset_name):
    result = {}
    dual_params = set(["sdf_ke", "sdf_kd", "sdf_kf", "sdf_mu",
                      "cut_spring_ke", "cut_spring_softness"])
    for key in datasets[dataset_name].params:
        if key.startswith("surface_") and key[len("surface_"):] in dual_params:
            result[key] = default_parameters[key[len("surface_"):]]
        else:
            result[key] = default_parameters[key]
    return result


if __name__ == "__main__":
    save_settings(default_settings, "config/default_cut_settings.json")
    # generate_configs(500, base_config_file="config/ansys_prism_potato.json", prefix="gen_2d_params/gen_2d_params")
    # generate_configs_sdf2d(500, base_config_file="config/ansys_prism_potato.json", prefix="gen_2d_sdf/gen_2d_sdf")
    for key, dataset in datasets.items():
        generate_configs(500, dataset.params, dataset.config_file, seed=123,
                         prefix=key, surface_equal_interior=dataset.surface_equal_interior)

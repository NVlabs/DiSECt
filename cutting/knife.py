# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import Enum


class KnifeType(Enum):
    YBJ = 1  # used in YBJ's work, our RSS paper
    EDC = 2  # small foldable EDC knife
    SLICING = 3  # large Victorinox 10" slicing knife


class Knife:
    def __init__(self, type: KnifeType):
        self.type = type
        if type == KnifeType.YBJ:
            self.spine_dim = 2e-3
            self.spine_height = 40e-3
            self.edge_dim = 0.08e-3
            self.tip_height = 0.04e-3
            self.depth = 150e-3
        elif type == KnifeType.EDC:
            self.spine_dim = 2e-3
            self.spine_height = 10e-3
            self.edge_dim = 0.08e-3
            self.tip_height = 0.04e-3
            self.depth = 65e-3
        if type == KnifeType.SLICING:
            self.spine_dim = 2e-3
            self.spine_height = 40e-3
            self.edge_dim = 0.08e-3
            self.tip_height = 0.04e-3
            self.depth = 254e-3

    def create_mesh(self):
        import numpy as np
        vertices = np.array([
            [-self.spine_dim / 2, self.spine_height / 2, -self.depth / 2],
            [self.spine_dim / 2, self.spine_height / 2, -self.depth / 2],
            [-self.edge_dim / 2, -self.spine_height / 2, -self.depth / 2],
            [self.edge_dim / 2, -self.spine_height / 2, -self.depth / 2],
            [0, -self.spine_height / 2 - self.tip_height, -self.depth / 2],
            [-self.spine_dim / 2, self.spine_height / 2, self.depth / 2],
            [self.spine_dim / 2, self.spine_height / 2, self.depth / 2],
            [-self.edge_dim / 2, -self.spine_height / 2, self.depth / 2],
            [self.edge_dim / 2, -self.spine_height / 2, self.depth / 2],
            [0, -self.spine_height / 2 - self.tip_height, self.depth / 2],
        ])

        # mesh faces
        faces = np.hstack([
            [0, 1, 3],
            [0, 3, 2],
            [2, 3, 4],
            [6, 5, 7],
            [6, 7, 8],
            [8, 7, 9],
            [1, 6, 8],
            [1, 8, 3],
            [3, 8, 9],
            [3, 9, 4],
            [0, 5, 6],
            [0, 6, 1],
            [5, 0, 2],
            [5, 2, 7],
            [7, 2, 4],
            [7, 4, 9],
        ])

        return vertices, faces

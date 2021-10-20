# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "meshcutter",
        ['meshing.cpp'],
        include_dirs=[],
        extra_compile_args=[],
        define_macros=[('VERSION_INFO', __version__)],
        cxx_std=17)
]

ParallelCompile("NPY_NUM_BUILD_JOBS",
                needs_recompile=naive_recompile).install()

setup(name='meshcutter',
      version=__version__,
      description='DiSECt mesh processing',
      long_description='Mesh preparation to add virtual nodes and cutting springs along a cutting surface represented by a triangle mesh',
      url='https://diff-cutting-sim.github.io',
      author='Eric Heiden',
      author_email='heiden@usc.edu',
      license='NVIDIA Source Code License',
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules
      )

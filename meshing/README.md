# Mesh cutting library

This C++ code implements our mesh processing algorithm which inserts virtual nodes into the mesh along the triangular cutting surface and connects these nodes via "cutting springs".

While DiSECt also provides a Python implementation of the same algorithm, we recommend using this C++ library for significantly improved performance (we observed more than 100x speed-ups for some meshes).

## Prerequisites

* Python 3.6 or higher
* C++17 compiler (e.g., MSVC 2019, gcc9 or higher, etc.)
* pybind11 Python bindings (can be installed via `pip install pybind11`)

## Installation

This library can be installed via either running
```sh
pip install -e . -v
```
or alternatively via
```
python setup.py install
```
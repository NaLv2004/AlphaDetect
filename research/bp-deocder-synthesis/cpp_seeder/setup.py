"""Build script for the pushgp_cpp pybind11 extension.

Build in-place:
    python setup.py build_ext --inplace

The module is installed as `pushgp._cpp_seeder` so the Python wrapper
`pushgp.cpp_seeder` can `from . import _cpp_seeder` after copying the
.pyd into the pushgp package directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = Path(__file__).resolve().parent

ext = Pybind11Extension(
    "pushgp_cpp_seeder",
    sources=[
        str(HERE / "src" / "bindings.cpp"),
    ],
    include_dirs=[str(HERE / "src")],
    cxx_std=17,
    define_macros=[("VERSION_INFO", '"0.1.0"')],
)

# MSVC-specific flags for performance
if sys.platform == "win32":
    ext.extra_compile_args.extend([
        "/O2", "/EHsc", "/MP", "/wd4244", "/wd4267",
    ])
else:
    ext.extra_compile_args.extend(["-O3", "-Wall", "-Wno-sign-compare"])

setup(
    name="pushgp_cpp_seeder",
    version="0.1.0",
    description="C++ accelerated VM/validator/RPG for Push-GP seeding",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

"""Build script for the pushgp_cpp_dce pybind11 extension.

Builds an independent .pyd named `pushgp_cpp_dce` that reuses
cpp_seeder's header-only VM / instruction / validator / opcodes (via
include paths) but produces a separate binary so the two modules can
coexist without ABI coupling.

Build in-place:
    python setup.py build_ext --inplace
"""
from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = Path(__file__).resolve().parent
SEEDER_SRC = HERE.parent / "cpp_seeder" / "src"

ext = Pybind11Extension(
    "pushgp_cpp_dce",
    sources=[
        str(HERE / "src" / "bindings.cpp"),
    ],
    include_dirs=[
        str(HERE / "src"),
        str(SEEDER_SRC),  # reuse vm.hpp / validator.hpp / opcodes.hpp / ...
    ],
    cxx_std=17,
    define_macros=[("VERSION_INFO", '"0.1.0"')],
)

if sys.platform == "win32":
    ext.extra_compile_args.extend([
        "/O2", "/EHsc", "/MP", "/wd4244", "/wd4267",
    ])
else:
    ext.extra_compile_args.extend(["-O3", "-Wall", "-Wno-sign-compare"])

setup(
    name="pushgp_cpp_dce",
    version="0.1.0",
    description="C++ accelerated DCE behavioral_reduce_bp (and BP decoder).",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

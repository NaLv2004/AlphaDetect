"""
Build C++ simulation code using MSVC (cl.exe) or MSBuild.

Usage:
    python build_cpp.py <source_dir> [options]

Options:
    --eigen <path>      Path to Eigen3 headers (default: mimo2D/eigen3)
    --output <path>     Output executable path
    --config <type>     Build configuration: Release or Debug (default: Release)
    --openmp            Enable OpenMP (default: True)
    --mkl               Link Intel MKL
    --sln <path>        Use MSBuild with a .sln file instead of cl.exe
    --platform <arch>   Target platform: x64 or x86 (default: x64)

Example:
    python build_cpp.py "research/topic/code/cpp/" --eigen "mimo2D/eigen3" --output "research/topic/code/cpp/sim.exe"
"""

import argparse
import glob
import os
import subprocess
import sys


def find_msbuild() -> str | None:
    """Find MSBuild.exe in common Visual Studio installation paths."""
    search_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\MSBuild\Current\Bin\MSBuild.exe",
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def find_vcvars64() -> str | None:
    """Find vcvars64.bat for MSVC environment setup."""
    search_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def build_with_msbuild(sln_path: str, config: str, platform: str) -> int:
    """Build using MSBuild."""
    msbuild = find_msbuild()
    if not msbuild:
        print("Error: MSBuild.exe not found.", file=sys.stderr)
        return 1

    cmd = [msbuild, sln_path, f"/p:Configuration={config}", f"/p:Platform={platform}", "/m"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def build_with_cl(source_dir: str, output: str, eigen_path: str | None,
                  config: str, openmp: bool, mkl: bool) -> int:
    """Build using cl.exe directly."""
    vcvars = find_vcvars64()
    if not vcvars:
        print("Error: vcvars64.bat not found.", file=sys.stderr)
        return 1

    # Collect all .cpp files
    cpp_files = glob.glob(os.path.join(source_dir, "*.cpp"))
    if not cpp_files:
        print(f"Error: No .cpp files found in {source_dir}", file=sys.stderr)
        return 1

    # Build cl.exe command
    flags = ["/EHsc", "/std:c++17"]
    if config == "Release":
        flags.extend(["/O2", "/DNDEBUG"])
    else:
        flags.extend(["/Od", "/Zi", "/D_DEBUG"])

    if openmp:
        flags.append("/openmp")

    includes = [f'/I"{source_dir}"']
    if eigen_path:
        includes.append(f'/I"{eigen_path}"')

    libs = []
    if mkl:
        libs.extend(["/link", "mkl_intel_lp64.lib", "mkl_sequential.lib", "mkl_core.lib"])

    file_list = ' '.join(f'"{f}"' for f in cpp_files)
    flag_str = ' '.join(flags)
    include_str = ' '.join(includes)
    output_flag = f'/Fe:"{output}"'

    # Build command (run through vcvars)
    build_cmd = f'"{vcvars}" && cl.exe {flag_str} {include_str} {file_list} {output_flag}'
    if libs:
        build_cmd += ' ' + ' '.join(libs)

    print(f"Building {len(cpp_files)} source files...")
    print(f"Output: {output}")
    result = subprocess.run(build_cmd, shell=True, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Build C++ simulation code.")
    parser.add_argument("source_dir", help="Directory containing source files")
    parser.add_argument("--eigen", help="Path to Eigen3 headers")
    parser.add_argument("--output", default="sim.exe", help="Output executable path")
    parser.add_argument("--config", choices=["Release", "Debug"], default="Release")
    parser.add_argument("--openmp", action="store_true", default=True)
    parser.add_argument("--mkl", action="store_true", default=False)
    parser.add_argument("--sln", help="Path to .sln file (use MSBuild instead of cl.exe)")
    parser.add_argument("--platform", choices=["x64", "x86"], default="x64")
    args = parser.parse_args()

    if args.sln:
        rc = build_with_msbuild(args.sln, args.config, args.platform)
    else:
        rc = build_with_cl(
            args.source_dir, args.output, args.eigen,
            args.config, args.openmp, args.mkl,
        )

    if rc == 0:
        print("Build successful.")
    else:
        print(f"Build failed with return code {rc}.", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()

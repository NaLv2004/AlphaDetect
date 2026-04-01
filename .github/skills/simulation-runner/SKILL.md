---
name: simulation-runner
description: "Build and run communications simulations (C++/Python/MATLAB). Use when: compiling C++ simulation code with MSBuild or cl.exe, running Monte Carlo simulations, executing parameter sweeps, collecting BER/FER/timing results, managing simulation execution on Windows."
argument-hint: "Describe the simulation to build and run, including path and parameters"
---

# Simulation Runner for Communications Research

## When to Use

- Building C++ simulation code (Visual Studio project or standalone)
- Running Monte Carlo simulations with parameter sweeps
- Executing Python or MATLAB simulation scripts
- Collecting and parsing simulation output

## Prerequisites

### C++ Build Tools
- **Visual Studio** (2017 or later) with C++ workload, OR
- **MSVC Build Tools** with `cl.exe` in PATH
- **Eigen3** library (included in workspace at `mimo2D/eigen3/`)

### Python
- Python 3.8+ with numpy, scipy (virtual environment in `.venv/`)

### MATLAB (optional)
- MATLAB R2020a+ with Communications Toolbox

## Procedure

### 1. Identify the Build Target

Check what type of project to build:
- **Visual Studio Solution** (`.sln`): Use MSBuild
- **CMakeLists.txt**: Use CMake + MSBuild
- **Standalone `.cpp` files**: Use `cl.exe` directly
- **Python scripts**: No build needed, check dependencies
- **MATLAB scripts**: No build needed

### 2. Build C++ Code

#### Option A: MSBuild (Visual Studio Solution)

```powershell
# Find MSBuild (adjust VS version as needed)
$msbuild = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "MSBuild.exe" | 
    Where-Object { $_.FullName -match "Current\\Bin\\MSBuild.exe" } | 
    Select-Object -First 1 -ExpandProperty FullName

& $msbuild "path\to\project.sln" /p:Configuration=Release /p:Platform=x64 /m
```

#### Option B: CMake

```powershell
$buildDir = "path\to\build"
New-Item -ItemType Directory -Force -Path $buildDir
Push-Location $buildDir
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
Pop-Location
```

#### Option C: Direct cl.exe (Standalone)

Use the build wrapper script:
```powershell
python "<skill-path>/scripts/build_cpp.py" "<source-dir>" --eigen "mimo2D/eigen3" --output "<output-exe>"
```

Or manually:
```powershell
# Set up MSVC environment
& "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"

cl.exe /EHsc /O2 /openmp /std:c++17 /I"mimo2D\eigen3" main.cpp /Fe:sim.exe
```

### 3. Run Simulation

Use the run script for parameter sweeps:
```powershell
python "<skill-path>/scripts/run_simulation.py" "<executable>" --snr-min 0 --snr-max 10 --snr-step 0.5 --output-dir "research/<topic>/results/raw/"
```

Or run directly:
```powershell
& "path\to\sim.exe" [arguments]
```

### 4. Collect Results

Parse output files into structured data:
```powershell
python "<skill-path>/scripts/collect_results.py" "research/<topic>/results/raw/" --output "research/<topic>/results/processed/results.csv"
```

### 5. Verify Results

- Check that BER/FER values are reasonable (monotonically decreasing with SNR)
- Verify enough frames were simulated (at least 100 error events per point)
- Compare against expected theoretical bounds
- Check for anomalies (sudden jumps, non-monotonic behavior)

## Build Configurations Reference

See [build-configs.md](./references/build-configs.md) for common build configurations.

# Build Configurations Reference

## Visual Studio Build (existing mimo2D project)

The existing codebase uses a Visual Studio 2017+ solution:
```
Solution: mimo2D/platform_1.sln
Project:  mimo2D/platform_1/platform_1.vcxproj
Target:   x64 Release
```

### Dependencies
- **Eigen3**: `mimo2D/eigen3/` (header-only, included via project settings)
- **OpenMP**: Enabled in project settings (`/openmp`)
- **Intel MKL**: Linked for BLAS/LAPACK (`/I"<MKL_PATH>/include"` + `mkl_intel_lp64.lib`)

### Build Command
```powershell
$msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
& $msbuild "mimo2D\platform_1.sln" /p:Configuration=Release /p:Platform=x64 /m
```

## Standalone C++ Build (new simulations)

For new simulation code that doesn't use a VS solution:

### Minimal (Eigen only)
```powershell
cl.exe /EHsc /O2 /std:c++17 /I"mimo2D\eigen3" main.cpp /Fe:sim.exe
```

### With OpenMP
```powershell
cl.exe /EHsc /O2 /openmp /std:c++17 /I"mimo2D\eigen3" *.cpp /Fe:sim.exe
```

### With MKL
```powershell
cl.exe /EHsc /O2 /openmp /std:c++17 /I"mimo2D\eigen3" /I"%MKLROOT%\include" ^
    *.cpp /Fe:sim.exe /link /LIBPATH:"%MKLROOT%\lib\intel64" ^
    mkl_intel_lp64.lib mkl_sequential.lib mkl_core.lib
```

### CMake Template
```cmake
cmake_minimum_required(VERSION 3.16)
project(CommSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../mimo2D/eigen3")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Sources
file(GLOB SOURCES "*.cpp")
add_executable(sim ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(sim OpenMP::OpenMP_CXX)
endif()
```

## Python Environment

```powershell
# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Install common dependencies
pip install numpy scipy matplotlib pandas
```

## MATLAB

```matlab
% Run from MATLAB command window or via command line:
matlab -batch "run('main_sim.m')"
```

cmake_minimum_required(VERSION 3.20.0)
project(cuMat)
# Main CMakeLists of cuMat

# Cuda is always required
cmake_policy(SET CMP0146 NEW)
find_package(CUDA REQUIRED)
add_definitions(-DEIGEN_NO_CUDA=1)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Add OpenMP if available. This is needed by cuSolver for optimal performance
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen is not required by default,
# but the tests depend on it -> use bundled Eigen for tests

# Hack: The FindCUDA that is used by the CI does not correctly support target_include_directories,
#  it does not forward the include paths.
#  Hence I have to include the paths globally.
include_directories(${CMAKE_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/third-party)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(third-party)
add_subdirectory(cuMat)
add_subdirectory(tests)
add_subdirectory(demos)
add_subdirectory(docs)
add_subdirectory(benchmarks)



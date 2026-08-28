cmake_minimum_required(VERSION 3.20)
project(Taichi_FMM LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(TAICHI_COMMON_CXX "-fopenmp -march=native -Wfatal-errors -Wno-format -fcx-limited-range")

# Host C++ Configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TAICHI_COMMON_CXX}")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -funroll-loops -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -funroll-loops -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG")

# Device CUDA Configuration
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${TAICHI_COMMON_CXX}\"")
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3 --use_fast_math -Xcompiler \"-O3 -funroll-loops -DNDEBUG\"")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo --use_fast_math -Xcompiler \"-O3 -g -DDEBUG\"")
set(CMAKE_CUDA_FLAGS_DEBUG          "-O0 -g -G -Xcompiler \"-O0 -g -DDEBUG\"")

set(TAICHI_CORE_DEFINITIONS
    FMM
    OUTPUTPOT
    OUTPUTACC
    EXPANSION=20
    H5_USE_16_API
)

add_executable(taichi_cpu
    src/cpu/main_cpu.cxx
)

target_include_directories(taichi_cpu BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/data_tables
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
    ${HDF5_INCLUDE_DIRS}
)

target_compile_definitions(taichi_cpu PRIVATE
    ${TAICHI_CORE_DEFINITIONS}
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:RelWithDebInfo>:DEBUG>
)

target_link_libraries(taichi_cpu PRIVATE
    OpenMP::OpenMP_CXX
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    z
    m
)

add_executable(taichi_gpu
    src/cuda/main_gpu.cxx
    src/cuda/EFMM.cu
)

target_include_directories(taichi_gpu BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/data_tables
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${HDF5_INCLUDE_DIRS}
)

target_compile_definitions(taichi_gpu PRIVATE
    ${TAICHI_CORE_DEFINITIONS}
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:RelWithDebInfo>:DEBUG>
)

target_link_libraries(taichi_gpu PRIVATE
    OpenMP::OpenMP_CXX
    ${HDF5_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    z
    m
    ${CMAKE_DL_LIBS}
)

# Pipeline Automation Targets
add_custom_target(verify
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pipeline_verify_profile.sh --verify
    DEPENDS taichi_cpu taichi_gpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Executing 10-step numerical parity check..."
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_target(profile
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pipeline_verify_profile.sh --profile
        DEPENDS taichi_cpu taichi_gpu
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Executing parity verification and generating NCU profile on 100k model..."
    )
else()
    add_custom_target(profile
        COMMAND ${CMAKE_COMMAND} -E false
        COMMENT "Profiling disabled in Debug mode"
    )
endif()

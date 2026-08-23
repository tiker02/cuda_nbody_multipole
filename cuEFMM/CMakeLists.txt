cmake_minimum_required(VERSION 3.18)

project(Taichi LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Find OpenMP for the C++ side
find_package(OpenMP REQUIRED)

# Define your executable and the source files it depends on
add_executable(Taichi 
    main.cxx 
    EFMM.cu
)

# ---------------------------------------------------------
# COMPILER FLAGS (Only applied to C++ files)
# ---------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -Wfatal-errors -O3 -Wno-format -march=native -fcx-limited-range")

# ---------------------------------------------------------
# PREPROCESSOR MACROS (-D flags)
# ---------------------------------------------------------
target_compile_definitions(Taichi PRIVATE
    FMM
    OUTPUTPOT
    OUTPUTACC
    EXPANSION=20
    DEBUG
    H5_USE_16_API
)

# ---------------------------------------------------------
# INCLUDES AND LIBRARIES (HDF5)
# ---------------------------------------------------------
# Where to find the headers (-I)
target_include_directories(Taichi PRIVATE 
    /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Where to find the libraries (-L)
target_link_directories(Taichi PRIVATE 
    /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib
)

# Which libraries to link against (-l)
target_link_libraries(Taichi PRIVATE 
    hdf5 
    z 
    m 
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
)
cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")

project(allreduce_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(CheckIPOSupported)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
check_ipo_supported(RESULT HAVE_LTO OUTPUT ERR_LTO)
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  if(HAVE_LTO)
    message (STATUS "LTO support found, enabling")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "No LTO: ${ERR_LTO}")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_executable(allreduce_bench)

target_sources(allreduce_bench PRIVATE
    bench.cpp
    memory.cu
    allreduce.cu
)

target_compile_options(allreduce_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -Wall,-Wextra>
)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(allreduce_bench PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
)

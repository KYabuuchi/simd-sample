cmake_minimum_required(VERSION 3.0)
project(simd-fir)

# Check C++17
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(
        FATAL_ERROR
            "${CMAKE_CXX_COMPILER} はC++17をサポートしてません。\n"
        )
endif()
set(CMAKE_CXX_STANDARD 17) # C++17
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")
message("\tusing C++17")

# Set default build type to Release
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# Optimize flag
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(OPTIMIZE_FLAGS "-mtune=native -march=native -mfpmath=both -Ofast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZE_FLAGSS}")
    message("Optimization flags:\n\t${OPTIMIZE_FLAGS}")
else()
    set(OPTIMIZE_FLAGS "")
    message("Not Optimized:\n\tBecause build type is ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -DENABLE_AVX512")

# OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV version:\n\t${OpenCV_VERSION}")

# =======================================
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} pthread)
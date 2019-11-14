cmake_minimum_required(VERSION 3.0)
project(simd-fir)

# Check C++17
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} はC++17をサポートしてません。\n")
endif()
set(CMAKE_CXX_STANDARD 17) # C++17
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")
message("\tusing C++17")

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# AVX512
if(NOT AVX512)
  set(AVX512 0)
endif()
message("AVX512:\n\t${AVX512}")

# Optimize flag
set(OPTIMIZE_FLAGS "-mtune=native -march=native -mfpmath=both -Ofast")
if(${AVX512})
  set(OPTIMIZE_FLAGS "${OPTIMIZE_FLAGS} -mfma -mavx512f -DENABLE_AVX512")
else()
  set(OPTIMIZE_FLAGS "${OPTIMIZE_FLAGS} -mfma -mavx -DENABLE_AVX")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZE_FLAGS}")
message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

# OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV version:\n\t${OpenCV_VERSION}")

# OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# =======================================
add_executable(inner inner.cpp)
target_link_libraries(inner ${OpenCV_LIBS} pthread)

add_executable(fir fir.cpp)
target_link_libraries(fir ${OpenCV_LIBS} pthread)

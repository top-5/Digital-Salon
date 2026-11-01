# CMake minimum version
cmake_minimum_required(VERSION 3.24)

# Project name
# Select CUDA before project() initializes the compiler. Override on other hosts.
if(UNIX AND NOT APPLE)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc CACHE FILEPATH "CUDA compiler")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
    endif()
endif()
project(DigitalSalon LANGUAGES CUDA CXX)
set(EXEC_NAME DigitalSalon)

# Check the operating system
if(WIN32)
    # Windows specific options
    message(STATUS "Compiling on Windows")
    include(cmake/windows.cmake)
    
elseif(UNIX AND NOT APPLE)
    # Linux specific options
    message(STATUS "Compiling on Linux")
    include(cmake/linux.cmake)

else()
    message(FATAL_ERROR "Unknown or unsupported operating system")
endif()

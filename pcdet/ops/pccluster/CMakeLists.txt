## Copyright (C) 2020  I. Bogoslavskyi, C. Stachniss
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.1)
project(depth_clustering)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  # set(CMAKE_BUILD_TYPE Debug)
endif()

# set(CMAKE_CXX_FLAGS "-Cluster -Wextra -fPIC")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

# cmake_policy(SET CMP0045 OLD)

message(STATUS ${PROJECT_SOURCE_DIR})
message(STATUS "BUILD HERE")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(Boost_USE_STATIC_LIBS OFF)

# option(BUILD_WITH_COVERAGE "Enable compilation with coverage information." OFF)
# if(BUILD_WITH_COVERAGE)
#   message(STATUS "Enabled generation of code coverage information with gcc.")
#   set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage ${CMAKE_CXX_FLAGS}")
# endif()
if(CMAKE_VERSION VERSION_LESS "3.15")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ListPrepend.cmake")
  list_prepend(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
else()
  list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

include(FetchContent)
include(pybind11)

# find_package(pybind11 REQUIRED)
# include_directories(${pybind11_INCLUDE_DIRS})

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(PCL QUIET COMPONENTS common io)
# find_package(QGLViewer REQUIRED)
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

message(STATUS "Found Open3D ${Open3D_VERSION}")

link_directories(${Open3D_LIBRARY_DIRS})


# add some useful constants to configuration
configure_file(config/cmake_config.h.in
               ${PROJECT_SOURCE_DIR}/config/cmake_config.h)
include_directories(config)


if (${PCL_FOUND})
  message(STATUS "PCL is found. PCL-related code will be built.")
  # if (${CMAKE_BUILD_TYPE} MATCHES "Debug" AND ${DISTRO} MATCHES "14.04")
  #   message(FATAL_ERROR
  #           "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
  #           "You are building PCL in Debug mode with C++11 enabled.\n"
  #           "This causes segmentation fault due to a bug in PCL.\n"
  #           "Build in Release mode to avoid this bug.\n"
  #           "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  # endif()
  add_definitions(-DPCL_FOUND)
else()
  message(WARNING "PCL NOT FOUND! PCL-related code WILL NOT be built")
endif()



include_directories(
  src
  # for QT moc files
  ${PROJECT_BINARY_DIR}/src
  # dependencies we have no control over
  SYSTEM ${EIGEN3_INCLUDE_DIR}
  SYSTEM ${Boost_INCLUDE_DIRS}
  SYSTEM ${PCL_INCLUDE_DIRS}
  SYSTEM ${OpenCV_INCLUDE_DIRS}
  # SYSTEM ${OpenGL_INCLUDE_DIRS}
  # SYSTEM ${QGLVIEWER_INCLUDE_DIR}
)

add_subdirectory(src)
# add_subdirectory(examples)
# add_subdirectory(clusteringprocessor)
add_subdirectory(python_wrapper)


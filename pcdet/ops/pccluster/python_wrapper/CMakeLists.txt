project(pccluster_python_wrapper)
find_package(Eigen3 REQUIRED QUIET)

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# set(PYBIND11_PYTHON_VERSION 3.7 CACHE STRING "")
# add_subdirectory(third_party/pybind11)
pybind11_add_module(pccluster pybinding.cpp)
# target_link_libraries(pccluster  Eigen3::Eigen)
target_link_libraries(pccluster PUBLIC clusteringprocessor)

# if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#   target_compile_options(pccluster PUBLIC -fsized-deallocation)
# endif()
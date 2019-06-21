# just here for CLion to work well
cmake_minimum_required(VERSION 3.14)

project (RynLib)
add_executable(RynLib RynLib.cpp)

set(MISC_DIR /Users/Mark/Documents/UW/Research/Common/private) # Just for CLion... don't use in practice
include_directories(${MISC_DIR})
#add_definitions(-DIM_A_REAL_BOY)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(RynLib ${PYTHON_LIBRARIES})

cmake_minimum_required(VERSION 3.14)
project(RynLib)

find_package(PythonLibs REQUIRED)
#find_package(Python3 REQUIRED NumPy)
#include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
# Fucking CMake always shits the bed...I thought the point was to _not_ have to hard code this kind of thing...
include_directories("/Users/Mark/Documents/Python/config/python3.7/lib/python3.7/site-packages/numpy/core/include")

add_compile_definitions(_OPENMP)
add_compile_definitions(_TBB)

#find_package(OpenMP REQUIRED)
#target_link_libraries(RynLib PRIVATE OpenMP::OpenMP_CXX)

#add_subdirectory("${PROJECT_SOURCE_DIR}/DoMyCode/src")
#add_subdirectory("${PROJECT_SOURCE_DIR}/Dumpi/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/PlzNumbers/src")
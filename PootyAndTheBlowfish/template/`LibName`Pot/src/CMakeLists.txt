# just here for CLion to work well
cmake_minimum_required(VERSION 3.14)

project (`LibName`)
set(
        SOURCE_FILES
        `LibName`.h
        `LibName`.cpp
        RynTypes.hpp
)
add_executable(`LibName` ${SOURCE_FILES})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(`LibName` ${PYTHON_LIBRARIES})

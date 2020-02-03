# just here for CLion to work well
cmake_minimum_required(VERSION 3.14)

project (RynLib)
set(
        SOURCE_FILES
        RynLib.h
        RynLib.cpp
            RynTypes.hpp
            Flargs.hpp
            MyPieIsGone.hpp
            MyPieIsGone.cpp
            MyPieIsHere.hpp
            MyPieIsHere.cpp
            PyAllUp.hpp
            PyAllUp.cpp
            Potators.hpp
            Potators.cpp
)
add_executable(RynLib ${SOURCE_FILES})

set(MISC_DIR /Users/Mark/Documents/UW/Research/Common/private/includes) # Just for CLion... don't use in practice
include_directories(${MISC_DIR})
#add_definitions(-DIM_A_REAL_BOY)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(RynLib ${PYTHON_LIBRARIES})

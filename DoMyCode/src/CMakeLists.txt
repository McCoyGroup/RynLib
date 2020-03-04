# just here for CLion to work well
cmake_minimum_required(VERSION 3.14)

project (DoMyCode)
set(
        SOURCE_FILES
        DoMyCode.h
        DoMyCode.cpp
            RynTypes.hpp
            WalkerPropagator.cpp
            WalkerPropagator.h
)
add_executable(DoMyCode ${SOURCE_FILES})

set(MISC_DIR /Users/Mark/Documents/UW/Research/Common/private/includes) # Just for CLion... don't use in practice
include_directories(${MISC_DIR})
#add_definitions(-DIM_A_REAL_BOY)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(DoMyCode ${PYTHON_LIBRARIES})

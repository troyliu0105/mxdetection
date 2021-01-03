cmake_minimum_required(VERSION 3.17)
project(cpp_extensions)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PythonInterp 3.0)

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import mxnet as mx;print(mx.libinfo.find_include_path())"
        OUTPUT_VARIABLE MXNET_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "Including MXNet: ${MXNET_INCLUDE}")
include_directories(${MXNET_INCLUDE})
add_library(corner_pooling SHARED corner_pooling.cpp)
add_library(iou_loss SHARED iou_loss.cpp)

set_target_properties(corner_pooling PROPERTIES SUFFIX ".so")
set_target_properties(iou_loss PROPERTIES SUFFIX ".so")
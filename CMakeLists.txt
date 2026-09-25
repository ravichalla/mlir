cmake_minimum_required(VERSION 3.20)
project(precision-lowering-pass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_library(PrecisionLoweringPass
    lib/PrecisionLoweringPass.cpp
)

target_link_libraries(PrecisionLoweringPass
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
)

add_executable(precision-lower-opt
    tools/precision-lower-opt.cpp
)

llvm_update_compile_flags(precision-lower-opt)

target_link_libraries(precision-lower-opt
  PRIVATE
    PrecisionLoweringPass
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
)

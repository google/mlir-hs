# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.13.4)

project(mlir_hs C CXX)

# Some niceties.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# CMake settings.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "We are actually building a static mondo-lib")
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL "Python soname linked libraries are bad")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

# Required LLVM settings.
set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "LLVM projects")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "Disable Z3")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Disable ZLIB")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "Only build for the host")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "Disable examples")
# TODO: MLIR is a "tool"
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "Disable tools")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "Disable tests")

# Required MLIR settings.
set(MLIR_BINDINGS_PYTHON_LOCK_VERSION OFF CACHE BOOL "Disable linking against libpython")

# LLVM configuration.
# Expected that LLVM is in/symlinked in to llvm-project.
set(LLVM_SRC_RELATIVE_DIR "llvm-project" CACHE STRING "Relative path to LLVM source")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_SRC_RELATIVE_DIR}/llvm" EXCLUDE_FROM_ALL)

add_mlir_aggregate(MLIRHaskell
  SHARED
  EMBED_LIBS
  MLIRCAPIConversion
  MLIRCAPIDebug
  MLIRCAPIExecutionEngine
  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRCAPITransforms

  # Dialects
  MLIRCAPIAsync
  MLIRCAPIGPU
  MLIRCAPILinalg
  MLIRCAPILLVM
  MLIRCAPIShape
  MLIRCAPISparseTensor
  MLIRCAPIStandard
  MLIRCAPISCF
  MLIRCAPITensor
)

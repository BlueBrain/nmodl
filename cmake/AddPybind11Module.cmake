# =============================================================================
# Copyright (C) 2020 Blue Brain Project
#
# See top-level LICENSE file for details.
# =============================================================================

include(FindPackageHandleStandardArgs)
find_package(FindPkgConfig QUIET)

find_path(
  PYBIND11_PROJ
  NAMES CMakeLists.txt
  PATHS "${PROJECT_SOURCE_DIR}/ext/pybind11")

find_package_handle_standard_args(PYBIND11 REQUIRED_VARS PYBIND11_PROJ)

if(NOT PYBIND11_FOUND)
  find_package(Git 1.8.3 QUIET)
  if(NOT ${GIT_FOUND})
    message(FATAL_ERROR "git not found, clone repository with --recursive")
  endif()
  message(STATUS "Sub-module pybind11 missing: running git submodule update --init --recursive")
  execute_process(
    COMMAND
      ${GIT_EXECUTABLE} submodule update --init --recursive --
      ${PROJECT_SOURCE_DIR}/ext/pybind11
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/ext/pybind11)

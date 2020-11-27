# =============================================================================
# Copyright (C) 2020 Blue Brain Project
#
# See top-level LICENSE file for details.
# =============================================================================

include(FindPackageHandleStandardArgs)
find_package(FindPkgConfig QUIET)

find_path(
  CLI11_PROJ
  NAMES CMakeLists.txt
  PATHS "${PROJECT_SOURCE_DIR}/ext/cli11")

find_package_handle_standard_args(CLI11 REQUIRED_VARS CLI11_PROJ)

if(NOT CLI11_FOUND)
  find_package(Git 1.8.3 QUIET)
  if(NOT ${GIT_FOUND})
    message(FATAL_ERROR "git not found, clone repository with --recursive")
  endif()
  message(STATUS "Sub-module CLI11 missing: running git submodule update --init --recursive")
  execute_process(
    COMMAND
      ${GIT_EXECUTABLE} submodule update --init --recursive --
      ${PROJECT_SOURCE_DIR}/ext/cli11
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/ext/cli11)

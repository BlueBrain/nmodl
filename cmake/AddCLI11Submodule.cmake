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

FIND_PACKAGE_HANDLE_STANDARD_ARGS(CLI11 REQUIRED_VARS CLI11_PROJ)

IF(NOT CLI11_FOUND)
  FIND_PACKAGE(GIT 1.8.3 QUIET)
  IF(NOT ${GIT_FOUND})
    MESSAGE(FATAL_ERROR "GIT NOT FOUND, CLONE REPOSITORY WITH --RECURSIVE")
  ENDIF()
  MESSAGE(STATUS "SUB-MODULE CLI11 MISSING: RUNNING GIT SUBMODULE UPDATE --INIT --RECURSIVE")
  EXECUTE_PROCESS(
    COMMAND
      ${GIT_EXECUTABLE} SUBMODULE UPDATE --INIT --RECURSIVE --
      ${PROJECT_SOURCE_DIR}/EXT/cli11
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
ENDIF()

Add_subdirectory(${PROJECT_SOURCE_DIR}/ext/cli11)

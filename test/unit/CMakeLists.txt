set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# =============================================================================
# Add extra compile flags to NMODL test sources
# =============================================================================
add_compile_options(${NMODL_EXTRA_CXX_FLAGS})
add_link_options(${NMODL_EXTRA_CXX_FLAGS})
add_compile_options(${NMODL_TESTS_COMPILER_WARNING_SUPPRESSIONS})

# =============================================================================
# Test executables
# =============================================================================
add_executable(
  testnmodl
  main.cpp
  codegen/codegen_compatibility_visitor.cpp
  codegen/codegen_coreneuron_cpp_visitor.cpp
  codegen/codegen_helper.cpp
  codegen/codegen_info.cpp
  codegen/codegen_neuron_cpp_visitor.cpp
  codegen/codegen_utils.cpp
  codegen/transform.cpp
  crout/crout.cpp
  ${CROUT_SOLVER_SOURCE_FILES}
  lexer/tokens.cpp
  modtoken/modtoken.cpp
  newton/newton.cpp
  ${NEWTON_SOLVER_SOURCE_FILES}
  parser/parser.cpp
  printer/printer.cpp
  symtab/symbol_table.cpp
  units/lexer.cpp
  units/parser.cpp
  utils/nmodl_constructs.cpp
  utils/string_utils.cpp
  utils/test_utils.cpp
  visitor/after_cvode_to_cnexp.cpp
  visitor/constant_folder.cpp
  visitor/defuse_analyze.cpp
  visitor/external.cpp
  visitor/global_to_range.cpp
  visitor/implicit_argument.cpp
  visitor/inline.cpp
  visitor/json.cpp
  visitor/kinetic_block.cpp
  visitor/local_to_assigned.cpp
  visitor/localize.cpp
  visitor/localrename.cpp
  visitor/lookup.cpp
  visitor/loop_unroll.cpp
  visitor/misc.cpp
  visitor/neuron_solve.cpp
  visitor/nmodl.cpp
  visitor/node_index.cpp
  visitor/perf.cpp
  visitor/rename.cpp
  visitor/rename_function_arguments.cpp
  visitor/semantic_analysis.cpp
  visitor/solve_block.cpp
  visitor/steadystate.cpp
  visitor/sympy_conductance.cpp
  visitor/sympy_solver.cpp
  visitor/units.cpp
  visitor/var_usage.cpp
  visitor/verbatim.cpp
  ${PROJECT_BINARY_DIR}/src/config/config.cpp)

target_include_directories(
  testnmodl
  PRIVATE ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${NMODL_PROJECT_SOURCE_DIR}/test
          ${NMODL_PROJECT_SOURCE_DIR}/src/solver ${NMODL_PROJECT_SOURCE_DIR}/src/utils
          ${NMODL_PROJECT_SOURCE_DIR}/ext/eigen)

target_link_libraries(
  testnmodl
  PUBLIC spdlog::spdlog
  PRIVATE visitor
          symtab
          lexer
          util
          printer
          codegen
          ${NMODL_WRAPPER_LIBS})

# =============================================================================
# Use catch_discover instead of add_test for granular test result reporting.
# =============================================================================
set(test_env ${NMODL_SANITIZER_ENABLE_ENVIRONMENT})
list(APPEND test_env "PYTHONPATH=${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH}")
if(NOT LINK_AGAINST_PYTHON)
  list(APPEND test_env "NMODL_PYLIB=$ENV{NMODL_PYLIB}")
endif()

# Without main from Catch2
target_link_libraries(testnmodl PRIVATE Catch2::Catch2)

set(CMAKE_REQUIRED_LIBRARIES Catch2::Catch2)
check_include_file_cxx("catch2/catch_version_macros.hpp" HAVE_CATCH2_V3)
unset(CMAKE_REQUIRED_LIBRARIES)
if(HAVE_CATCH2_V3)
  target_compile_definitions(testnmodl PRIVATE NMODL_CATCH2_VERSION_MAJOR=3)
else()
  target_compile_definitions(testnmodl PRIVATE NMODL_CATCH2_VERSION_MAJOR=2)
endif()

cpp_cc_configure_sanitizers(TARGET testnmodl)
# See https://github.com/catchorg/Catch2/issues/2424
string(REPLACE ";" ";ENVIRONMENT;" _environment_vars_list "${test_env}")
# Catch2 doesn't set these environment variables when running the executable to discover tests
# (https://github.com/catchorg/Catch2/issues/1810). This means that in builds that enable it, we see
# errors from the LeakSanitizer during the build. These seem to be harmless because Catch2 does not
# seem to check the exit code.
catch_discover_tests(testnmodl TEST_PREFIX "testnmodl/" PROPERTIES ENVIRONMENT
                     "${_environment_vars_list}")

# Generate include files used in test input, see test/unit/utils/nmodl_constructs.cpp
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/Unit.inc" "UNITSON \n UNITSOFF \n UNITSON \n UNITSOFF")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/included.file" "TITLE")

if(WIN32)
  set(NMODL_TEST_PYTHONPATH "${PROJECT_BINARY_DIR}/lib;$ENV{PYTHONPATH}")
else()
  set(NMODL_TEST_PYTHONPATH "${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH}")
endif()

# =============================================================================
# pybind11 tests
# =============================================================================
add_test(NAME Ode COMMAND ${PYTHON_EXECUTABLE} -m pytest ${NMODL_PROJECT_SOURCE_DIR}/test/unit/ode)
set_tests_properties(Ode PROPERTIES ENVIRONMENT "PYTHONPATH=${NMODL_TEST_PYTHONPATH}")

if(NMODL_ENABLE_PYTHON_BINDINGS)
  add_test(NAME Pybind COMMAND ${PYTHON_EXECUTABLE} -m pytest
                               ${NMODL_PROJECT_SOURCE_DIR}/test/unit/pybind)
  set_tests_properties(Pybind PROPERTIES ENVIRONMENT "PYTHONPATH=${NMODL_TEST_PYTHONPATH}")
  cpp_cc_configure_sanitizers(TEST Pybind PRELOAD)
endif()

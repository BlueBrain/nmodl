# =============================================================================
# llvm benchmark sources
# =============================================================================
set(LLVM_BENCHMARK_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_benchmark.cpp ${CMAKE_CURRENT_SOURCE_DIR}/llvm_benchmark.hpp)

# =============================================================================
# LLVM benchmark library
# =============================================================================
include_directories(${LLVM_INCLUDE_DIRS})
add_library(llvm_benchmark STATIC ${LLVM_BENCHMARK_SOURCE_FILES})
add_dependencies(llvm_benchmark lexer util visitor)

if(NMODL_ENABLE_JIT_EVENT_LISTENERS)
  target_compile_definitions(llvm_benchmark PUBLIC NMODL_HAVE_JIT_EVENT_LISTENERS)
endif()

# =============================================================================
# LLVM pyjit
# =============================================================================

if(NMODL_ENABLE_PYTHON_BINDINGS)
  file(GLOB modfiles "${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/kernels/*.mod")
  foreach(modfile ${modfiles})
    get_filename_component(modfile_name "${modfile}" NAME)
    add_test(NAME "PyJIT/${modfile_name}"
             COMMAND ${PYTHON_EXECUTABLE} ${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/benchmark.py
                     ${modfile})
    set_tests_properties(
      "PyJIT/${modfile_name}" PROPERTIES ENVIRONMENT
                                         PYTHONPATH=${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH})
  endforeach()
endif()

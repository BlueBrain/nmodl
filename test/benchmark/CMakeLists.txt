# =============================================================================
# llvm benchmark sources
# =============================================================================
set(LLVM_BENCHMARK_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_benchmark.cpp ${CMAKE_CURRENT_SOURCE_DIR}/llvm_benchmark.hpp)

if(NMODL_ENABLE_LLVM_CUDA)
  list(APPEND LLVM_BENCHMARK_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_driver.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cuda_driver.hpp)
endif()

# =============================================================================
# LLVM benchmark library
# =============================================================================
include_directories(${LLVM_INCLUDE_DIRS})
add_library(llvm_benchmark STATIC ${LLVM_BENCHMARK_SOURCE_FILES})
add_dependencies(llvm_benchmark lexer util visitor)
if(NMODL_ENABLE_LLVM_CUDA)
  target_link_libraries(llvm_benchmark PRIVATE CUDA::cudart CUDA::nvrtc)
endif()

if(NMODL_ENABLE_JIT_EVENT_LISTENERS)
  target_compile_definitions(llvm_benchmark PUBLIC NMODL_HAVE_JIT_EVENT_LISTENERS)
endif()

# =============================================================================
# LLVM pyjit
# =============================================================================

if(NMODL_ENABLE_PYTHON_BINDINGS)
  file(GLOB modfiles "${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/kernels/*.mod")
  list(APPEND modfiles "${NMODL_PROJECT_SOURCE_DIR}/test/integration/mod/test_math.mod")
  foreach(modfile ${modfiles})
    # For expsyn.mod set the vector width to 1 since atomic operations are not supported for vector
    # widths > 1. See https://github.com/BlueBrain/nmodl/issues/857
    if(${modfile} STREQUAL "${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/kernels/expsyn.mod")
      set(extra_args "--vec 1")
    endif()
    get_filename_component(modfile_name "${modfile}" NAME)
    add_test(NAME "PyJIT/${modfile_name}"
             COMMAND ${PYTHON_EXECUTABLE} ${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/benchmark.py
                     --file ${modfile} ${extra_args})
    set_tests_properties(
      "PyJIT/${modfile_name}" PROPERTIES ENVIRONMENT
                                         PYTHONPATH=${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH})

    if(NMODL_ENABLE_LLVM_CUDA)
      add_test(NAME "PyJIT/${modfile_name}_gpu"
               COMMAND ${PYTHON_EXECUTABLE} ${NMODL_PROJECT_SOURCE_DIR}/test/benchmark/benchmark.py
                      --file ${modfile} --gpu ${extra_args})
      message(STATUS "CUDA_HOME is ${CUDAToolkit_TARGET_DIR}")
      set_tests_properties(
        "PyJIT/${modfile_name}_gpu"
        PROPERTIES
          ENVIRONMENT
          "PYTHONPATH=${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH};CUDA_HOME=${CUDAToolkit_TARGET_DIR}"
      )
    endif()
  endforeach()
endif()

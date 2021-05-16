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

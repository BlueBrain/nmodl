# =============================================================================
# Codegen sources
# =============================================================================
set(LLVM_CODEGEN_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_debug_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_debug_builder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_ir_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm_ir_builder.hpp)

# =============================================================================
# LLVM codegen library and executable
# =============================================================================

include_directories(${LLVM_INCLUDE_DIRS})
add_library(llvm_codegen OBJECT ${LLVM_CODEGEN_SOURCE_FILES})
set_property(TARGET llvm_codegen PROPERTY POSITION_INDEPENDENT_CODE ON)

add_dependencies(llvm_codegen lexer util visitor)
target_link_libraries(llvm_codegen PRIVATE util)

if(NOT NMODL_AS_SUBPROJECT)
  add_executable(nmodl_llvm_runner main.cpp)

  target_link_libraries(
    nmodl_llvm_runner
    CLI11::CLI11
    llvm_benchmark
    llvm_codegen
    codegen
    visitor
    symtab
    lexer
    util
    test_util
    printer
    ${NMODL_WRAPPER_LIBS}
    ${LLVM_LIBS_TO_LINK})
endif()

# =============================================================================
# Install executable
# =============================================================================

if(NOT NMODL_AS_SUBPROJECT)
  install(TARGETS nmodl_llvm_runner DESTINATION ${NMODL_INSTALL_DIR_SUFFIX}bin)
endif()

# =============================================================================
# Codegen sources
# =============================================================================
set(LLVM_CODEGEN_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.hpp)

# =============================================================================
# LLVM codegen library
# =============================================================================

include_directories(${LLVM_INCLUDE_DIRS})
add_library(llvm_codegen STATIC ${LLVM_CODEGEN_SOURCE_FILES})
add_dependencies(llvm_codegen lexer util visitor)

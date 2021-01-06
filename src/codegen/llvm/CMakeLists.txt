# =============================================================================
# Codegen sources
# =============================================================================
set(LLVM_CODEGEN_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_visitor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen_llvm_helper_visitor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jit_driver.hpp)

# =============================================================================
# LLVM codegen library and executable
# =============================================================================

include_directories(${LLVM_INCLUDE_DIRS})
add_library(runner_obj OBJECT ${LLVM_CODEGEN_SOURCE_FILES})
add_dependencies(runner_obj lexer_obj)
set_property(TARGET runner_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(llvm_codegen STATIC $<TARGET_OBJECTS:runner_obj>)

add_dependencies(llvm_codegen lexer util visitor)

if(NOT NMODL_AS_SUBPROJECT)
  add_executable(nmodl_llvm_runner main.cpp)

  target_link_libraries(
    nmodl_llvm_runner
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

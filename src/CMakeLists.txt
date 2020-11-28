# =============================================================================
# Add extra compile flags to NMODL sources
# =============================================================================
add_compile_options(${NMODL_EXTRA_CXX_FLAGS})
add_link_options(${NMODL_EXTRA_CXX_FLAGS})

add_subdirectory(codegen)
add_subdirectory(language)
add_subdirectory(lexer)
add_subdirectory(parser)
add_subdirectory(printer)
add_subdirectory(symtab)
add_subdirectory(utils)
add_subdirectory(visitors)
add_subdirectory(pybind)
add_subdirectory(solver)

# =============================================================================
# NMODL sources
# =============================================================================
include_directories(${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

# =============================================================================
# Add executables
# =============================================================================
add_executable(nmodl main.cpp)
target_link_libraries(
  nmodl
  CLI11::CLI11
  printer
  codegen
  visitor
  symtab
  util
  lexer
  ${NMODL_WRAPPER_LIBS})
cpp_cc_configure_sanitizers(TARGET nmodl)

if(NMODL_ENABLE_LLVM)
  # LLVM core libraries to link
  llvm_map_components_to_libnames(LLVM_CORE_LIBS core)
  target_link_libraries(nmodl llvm_codegen ${LLVM_CORE_LIBS})
endif()

# =============================================================================
# Add dependency with nmodl pytnon module (for consumer projects)
# =============================================================================
add_dependencies(nmodl pywrapper)

if(NMODL_ENABLE_PYTHON_BINDINGS)
  add_dependencies(nmodl _nmodl)
endif()

# =============================================================================
# Install executable
# =============================================================================
install(TARGETS nmodl DESTINATION ${NMODL_INSTALL_DIR_SUFFIX}bin)
install(FILES nmodl.hpp DESTINATION ${NMODL_INSTALL_DIR_SUFFIX}include)

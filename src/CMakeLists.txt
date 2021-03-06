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
set(NMODL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# =============================================================================
# Add executables
# =============================================================================
add_executable(nmodl ${NMODL_SOURCE_FILES})
target_link_libraries(
  nmodl
  printer
  codegen
  visitor
  symtab
  util
  lexer
  ${NMODL_WRAPPER_LIBS})

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

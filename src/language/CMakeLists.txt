# =============================================================================
# Command to generate AST/Visitor classes from language definition
# =============================================================================
set_source_files_properties(${NMODL_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

add_custom_command(
  OUTPUT ${NMODL_GENERATED_SOURCES}
  COMMAND ${PYTHON_EXECUTABLE} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/code_generator.py --base-dir
          ${PROJECT_BINARY_DIR}/src
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CODE_GENERATOR_PY_FILES}
  DEPENDS ${CODE_GENERATOR_YAML_FILES}
  DEPENDS ${CODE_GENERATOR_JINJA_FILES}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/templates/code_generator.cmake
  COMMENT "-- NMODL : GENERATING AST CLASSES WITH PYTHON GENERATOR! --")

# =============================================================================
# Target to propagate dependencies properly to lexer
# =============================================================================
add_custom_target(pyastgen DEPENDS ${PROJECT_BINARY_DIR}/src/ast/ast.cpp)

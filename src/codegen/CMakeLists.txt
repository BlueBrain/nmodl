# =============================================================================
# Codegen library
# =============================================================================
add_library(
  codegen STATIC
  codegen_acc_visitor.cpp
  codegen_transform_visitor.cpp
  codegen_c_visitor.cpp
  codegen_compatibility_visitor.cpp
  codegen_helper_visitor.cpp
  codegen_info.cpp
  codegen_utils.cpp)
add_dependencies(codegen lexer util visitor)
target_link_libraries(codegen PRIVATE util)

# copy to build directory to make usable from build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fast_math.hpp
               ${CMAKE_BINARY_DIR}/include/nmodl/fast_math.hpp COPYONLY)
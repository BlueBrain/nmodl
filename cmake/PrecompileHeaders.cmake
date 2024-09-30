function(target_precompile_nmodl_headers target_name)

  if(COMMAND target_precompile_headers)
    target_precompile_headers(${target_name}
      # PRIVATE <string>
      # PRIVATE <filesystem>
      PRIVATE <utils/logger.hpp>
      PRIVATE <ast/ast.hpp>
      PRIVATE <ast/all.hpp>)
  else()
    message(WARNING "CMake doesn't support `target_precompile_headers`.")
  endif()
endfunction()

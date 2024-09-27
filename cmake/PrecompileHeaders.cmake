function(target_precompile_nmodl_headers target_name)
  target_precompile_headers(${target_name}
    PRIVATE <string>
    PRIVATE <filesystem>
    PRIVATE <utils/logger.hpp>
    PRIVATE <ast/ast.hpp>
    PRIVATE <ast/all.hpp>)
endfunction()

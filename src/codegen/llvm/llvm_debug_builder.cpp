/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/llvm_debug_builder.hpp"

namespace nmodl {
namespace codegen {


static constexpr const char debug_version_key[] = "Debug Version";


void DebugBuilder::add_function_debug_info(llvm::Function* function, Location* loc) {
    // Create the function debug type (subroutine type). We are not interested in parameters and
    // types, and therefore passing llvm::None as argument suffices for now.
    llvm::DISubroutineType* subroutine_type = di_builder.createSubroutineType(
        di_builder.getOrCreateTypeArray(llvm::None));
    llvm::DISubprogram::DISPFlags sp_flags = llvm::DISubprogram::SPFlagDefinition |
                                             llvm::DISubprogram::SPFlagOptimized;
    // If there is no location associated with the function, just use 0.
    int line = loc ? loc->line : 0;
    llvm::DISubprogram* program = di_builder.createFunction(compile_unit,
                                                            function->getName(),
                                                            function->getName(),
                                                            file,
                                                            line,
                                                            subroutine_type,
                                                            line,
                                                            llvm::DINode::FlagZero,
                                                            sp_flags);
    function->setSubprogram(program);
    di_builder.finalizeSubprogram(program);
}

void DebugBuilder::create_compile_unit(llvm::Module& module,
                                       const std::string& debug_filename,
                                       const std::string& debug_output_dir) {
    // Create the debug file and compile unit for the module.
    file = di_builder.createFile(debug_filename, debug_output_dir);
    compile_unit = di_builder.createCompileUnit(llvm::dwarf::DW_LANG_C,
                                                file,
                                                /*Producer=*/"NMODL-LLVM",
                                                /*isOptimized=*/false,
                                                /*Flags=*/"",
                                                /*RV=*/0);

    // Add a flag to the module to specify that it has debug information.
    if (!module.getModuleFlag(debug_version_key)) {
        module.addModuleFlag(llvm::Module::Warning,
                             debug_version_key,
                             llvm::DEBUG_METADATA_VERSION);
    }
}

void DebugBuilder::finalize() {
    di_builder.finalize();
}
}  // namespace codegen
}  // namespace nmodl

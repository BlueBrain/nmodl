/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace nmodl {
namespace codegen {

/// A struct to store AST location information.
/// \todo Currently, not all AST nodes have location information. Moreover,
/// some may not have it as they were artificially introduced (e.g.
/// CodegenForStatement). This simple wrapper suffices for now, but in future
/// we may want to handle this properly.
struct Location {
    /// Line in the file.
    int line;

    /// Column in the file.
    int column;
};


/**
 * \class DebugBuilder
 * \brief A helper class to create debug information for LLVM IR module.
 * \todo Only function debug information is supported.
 */
class DebugBuilder {
  private:
    /// Debug information builder.
    llvm::DIBuilder di_builder;

    /// LLVM context.
    llvm::LLVMContext& context;

    /// Debug compile unit for the module.
    llvm::DICompileUnit* compile_unit = nullptr;

    /// Debug file pointer.
    llvm::DIFile* file = nullptr;

  public:
    DebugBuilder(llvm::Module& module)
        : di_builder(module)
        , context(module.getContext()) {}

    /// Adds function debug information with an optional location.
    void add_function_debug_info(llvm::Function* function, Location* loc = nullptr);

    /// Creates the compile unit for and sets debug flags for the module.
    void create_compile_unit(llvm::Module& module,
                             const std::string& debug_filename,
                             const std::string& debug_output_dir);

    /// Finalizes the debug information.
    void finalize();
};
}  // namespace codegen
}  // namespace nmodl

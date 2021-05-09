/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"
#include "symtab/symbol_table.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

namespace nmodl {
namespace codegen {

/// Floating point bit widths.
static constexpr const unsigned single_precision= 32;
static constexpr const unsigned double_precision= 64;

/**
 * \class IRBuilder
 * \brief A helper class to generate LLVM IR for NMODL AST.
 */
class IRBuilder {
  private:
    /// Underlying LLVM IR builder.
    llvm::IRBuilder<> builder;

    /// Symbol table of the NMODL AST.
    symtab::SymbolTable* symbol_table;

    /// Pointer to the current function for which the code is generated.
    llvm::Function* current_function;

    /// Stack to hold visited and processed values.
    std::vector<llvm::Value*> value_stack;

    /// Flag to indicate that the generated IR should be vectorized.
    bool vectorize;

    /// A helper to query the instance variable information.
    InstanceVarHelper* instance_var_info;

    /// Precision of the floating-point numbers (32 or 64 bit).
    unsigned fp_precision;

    /// The vector width if generating vectorized code.
    unsigned vector_width;

    /// The name of induction variable used in kernel loops.
    const std::string kernel_id;

  public:
    IRBuilder(llvm::LLVMContext &context,
              symtab::SymbolTable* symbol_table,
              InstanceVarHelper* instance_var_info,
              unsigned fp_precision = double_precision,
              unsigned vector_width = 1,
              const std::string kernel_id = "id")
        : builder(context)
        , symbol_table(symbol_table)
        , current_function(nullptr)
        , vectorize(false)
        , instance_var_info(instance_var_info)
        , fp_precision(fp_precision)
        , vector_width(vector_width)
        , kernel_id(kernel_id) {}

  private:


};
}  // namespace codegen
}  // namespace nmodl

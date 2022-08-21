/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/base_builder.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

namespace nmodl {
namespace codegen {

/**
 * \class GPUBuilder
 * \brief A class to generate LLVM IR for NMODL AST targeting GPU platforms.
 */
class GPUBuilder: public BaseBuilder {
  protected:
     GPUBuilder(llvm::LLVMContext& context,
                 bool single_precision,
                 std::vector<std::string> fast_math_flags = {})
        : BaseBuilder(context, single_precision, fast_math_flags) {}

    /*************************************************************************/
    /*                     Virtual generation methods                        */
    /*************************************************************************/

  public:
    /// Generates LLVM IR to handle atomic updates, e.g. *ptr += rhs.
    void generate_atomic_statement(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op) override;

    /// Generates LLVM IR for loop initialization.
    void generate_loop_start() override;

    /// Generates LLVM IR for loop increment.
    void generate_loop_increment() override;

};

}  // namespace codegen
}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief LLVM based code generation backend implementation for CoreNEURON
 *
 * \file
 * \brief \copybrief nmodl::codegen::CodegenLLVMVisitor
 */

#include <ostream>
#include <string>

#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace codegen {

/**
 * @defgroup llvm LLVM Based Code Generation Implementation
 * @brief Implementations of LLVM based code generation
 *
 * @defgroup llvm_backends LLVM Codegen Backend
 * @ingroup llvm
 * @brief Code generation backends for NMODL AST to LLVM IR
 * @{
 */

/**
 * \class CodegenLLVMVisitor
 * \brief %Visitor for transforming NMODL AST to LLVM IR
 */
class CodegenLLVMVisitor: public visitor::ConstAstVisitor {
    // Name of mod file (without .mod suffix)
    std::string mod_filename;

    // Output directory for code generation
    std::string output_dir;

    // result string for demo
    std::string result_code;

  public:
    /**
     * \brief Constructs the LLVM code generator visitor
     *
     * This constructor instantiates an NMODL LLVM code generator. This is
     * just template to work with initial implementation.
     */
    CodegenLLVMVisitor(const std::string& mod_filename, const std::string& output_dir)
        : mod_filename(mod_filename)
        , output_dir(output_dir) {}

    void visit_statement_block(const ast::StatementBlock& node) override;
    void visit_procedure_block(const ast::ProcedureBlock& node) override;
    void visit_program(const ast::Program& node) override;

    // demo method
    std::string get_code() const {
        return result_code;
    }
};

/** \} */  // end of llvm_backends

}  // namespace codegen
}  // namespace nmodl

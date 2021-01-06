/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::codegen::CodegenLLVMHelperVisitor
 */

#include <string>

#include "codegen/codegen_info.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace codegen {

/**
 * @addtogroup llvm_codegen_details
 * @{
 */

/**
 * \class CodegenLLVMHelperVisitor
 * \brief Helper visitor to gather AST information to help LLVM code generation
 */
class CodegenLLVMHelperVisitor: public visitor::AstVisitor {
    std::vector<std::shared_ptr<ast::CodegenFunction>> codegen_functions;

    void add_function_procedure_node(ast::Block& node);

  public:
    CodegenLLVMHelperVisitor() = default;

    void visit_statement_block(ast::StatementBlock& node) override;
    void visit_procedure_block(ast::ProcedureBlock& node) override;
    void visit_function_block(ast::FunctionBlock& node) override;
    void visit_program(ast::Program& node) override;
};

/** @} */  // end of llvm_codegen_details

}  // namespace codegen
}  // namespace nmodl

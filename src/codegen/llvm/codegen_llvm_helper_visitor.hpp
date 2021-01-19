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


typedef std::vector<std::shared_ptr<ast::CodegenFunction>> CodegenFunctionVector;

/**
 * @addtogroup llvm_codegen_details
 * @{
 */

/**
 * \class CodegenLLVMHelperVisitor
 * \brief Helper visitor for AST information to help code generation backends
 *
 * Code generation backends convert NMODL AST to C++ code. But during this
 * C++ code generation, various transformations happens and final code generated
 * is quite different / large than actual kernel represented in MOD file ro
 * NMODL AST.
 *
 * Currently, these transformations are embedded into code generation backends
 * like ast::CodegenCVisitor. If we have to generate code for new simulator, there
 * will be duplication of these transformations. Also, for completely new
 * backends like NEURON simulator or SIMD library, we will have code duplication.
 *
 * In order to avoid this, we perform maximum transformations in this visitor.
 * Currently we focus on transformations that will help LLVM backend but later
 * these will be common across all backends.
 */
class CodegenLLVMHelperVisitor: public visitor::AstVisitor {
    /// newly generated code generation specific functions
    CodegenFunctionVector codegen_functions;

    /// ast information for code generation
    codegen::CodegenInfo info;

    /// default integer and float node type
    const ast::AstNodeType INTEGER_TYPE = ast::AstNodeType::INTEGER;
    const ast::AstNodeType FLOAT_TYPE = ast::AstNodeType::DOUBLE;

    /// create new function for FUNCTION or PROCEDURE block
    void create_function_for_node(ast::Block& node);

    /// create new LLVMStructBlock
    std::shared_ptr<ast::LLVMStructBlock> create_llvm_struct_block();

  public:
    CodegenLLVMHelperVisitor() = default;

    /// run visitor and return code generation functions
    CodegenFunctionVector get_codegen_functions(const ast::Program& node);

    void ion_read_statements(BlockType type,
                             std::vector<std::string>& int_variables,
                             std::vector<std::string>& double_variables,
                             ast::StatementVector& index_statements,
                             ast::StatementVector& body_statements);

    void ion_write_statements(BlockType type,
                              std::vector<std::string>& int_variables,
                              std::vector<std::string>& double_variables,
                              ast::StatementVector& index_statements,
                              ast::StatementVector& body_statements);

    void convert_to_instance_variable(ast::Node& node, std::string& index_var);

    void convert_local_statement(ast::StatementBlock& node);

    void visit_procedure_block(ast::ProcedureBlock& node) override;
    void visit_function_block(ast::FunctionBlock& node) override;
    void visit_nrn_state_block(ast::NrnStateBlock& node) override;
    void visit_program(ast::Program& node) override;
};

/** @} */  // end of llvm_codegen_details

}  // namespace codegen
}  // namespace nmodl

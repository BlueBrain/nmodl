
/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen_llvm_helper_visitor.hpp"

#include "ast/all.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace codegen {

using namespace fmt::literals;

void CodegenLLVMHelperVisitor::visit_statement_block(ast::StatementBlock& node) {
    node.visit_children(*this);

    /// if local list statement exist, we have to replace it
    auto local_statement = visitor::get_local_list_statement(node);
    if (local_statement) {
        /// create codegen variables from local variables
        ast::CodegenVarVector variables;
        for (const auto& var: local_statement->get_variables()) {
            variables.emplace_back(new ast::CodegenVar(0, var->get_name()->clone()));
        }

        /// remove local list statement now
        const auto& statements = node.get_statements();
        node.erase_statement(statements.begin());

        /// create new codegen variable statement
        auto type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
        auto statement = std::make_shared<ast::CodegenVarListStatement>(type, variables);

        /// insert codegen variable statement
        node.insert_statement(statements.begin(), statement);
    }
}

void CodegenLLVMHelperVisitor::add_function_procedure_node(ast::Block& node) {
    std::string function_name = node.get_node_name();

    const auto& source_node_type = node.get_node_type();
    auto name = new ast::Name(new ast::String(function_name));
    auto return_var = new ast::Name(new ast::String("ret_" + function_name));
    ast::CodegenVarType* var_type = nullptr;
    ast::CodegenVarType* return_type = nullptr;

    /// return type based on node type
    bool is_function = source_node_type == ast::AstNodeType::FUNCTION_BLOCK;
    if (is_function) {
        var_type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
    } else {
        var_type = new ast::CodegenVarType(ast::AstNodeType::INTEGER);
    }

    /// return type is same as variable type
    return_type = var_type->clone();

    /// function body and it's statement
    auto block = node.get_statement_block()->clone();
    const auto& statements = block->get_statements();

    /// insert return variable at the start of the block
    ast::CodegenVarVector codegen_vars;
    codegen_vars.emplace_back(new ast::CodegenVar(0, return_var->clone()));
    auto statement = std::make_shared<ast::CodegenVarListStatement>(var_type, codegen_vars);
    block->insert_statement(statements.begin(), statement);

    /// add return statement
    auto return_statement = new ast::CodegenReturnStatement(return_var);
    block->emplace_back_statement(return_statement);

    /// prepare arguments
    ast::CodegenArgumentVector code_arguments;
    const auto& arguments = node.get_parameters();
    for (const auto& arg: arguments) {
        auto type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
        auto var = arg->get_name()->clone();
        code_arguments.emplace_back(new ast::CodegenArgument(type, var));
    }

    /// add new node to AST
    auto function =
        std::make_shared<ast::CodegenFunction>(return_type, name, code_arguments, block);
    codegen_functions.push_back(function);
}

void CodegenLLVMHelperVisitor::visit_procedure_block(ast::ProcedureBlock& node) {
    node.visit_children(*this);
    add_function_procedure_node(node);
}

void CodegenLLVMHelperVisitor::visit_function_block(ast::FunctionBlock& node) {
    node.visit_children(*this);
    add_function_procedure_node(node);
}

void CodegenLLVMHelperVisitor::visit_program(ast::Program& node) {
    logger->info("Running CodegenLLVMHelperVisitor");
    node.visit_children(*this);
    for (auto& fun: codegen_functions) {
        node.emplace_back_node(fun);
    }
}

}  // namespace codegen
}  // namespace nmodl

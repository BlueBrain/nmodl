/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/derivative_original_visitor.hpp"

#include "ast/all.hpp"
#include "pybind/pyembed.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace pywrap = nmodl::pybind_wrappers;

namespace nmodl {
namespace visitor {


void DerivativeOriginalVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    // TODO figure out what happens when we have both a KINETIC and a DERIVATIVE block in the same
    // mod file
    node.visit_children(*this);
    der_block_function = node.clone();
    der_block_jacobian = node.clone();
}


void DerivativeOriginalVisitor::visit_derivative_original_function_block(
    ast::DerivativeOriginalFunctionBlock& node) {
    derivative_block = true;
    node_type = node.get_node_type();
    node.visit_children(*this);
    node_type = ast::AstNodeType::NODE;
    derivative_block = false;
}

void DerivativeOriginalVisitor::visit_derivative_original_jacobian_block(
    ast::DerivativeOriginalJacobianBlock& node) {
    derivative_block = true;
    node_type = node.get_node_type();
    node.visit_children(*this);
    node_type = ast::AstNodeType::NODE;
    derivative_block = false;
}

void DerivativeOriginalVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    differential_equation = true;
    node.visit_children(*this);
    differential_equation = false;
}


void DerivativeOriginalVisitor::visit_binary_expression(ast::BinaryExpression& node) {
    const auto& lhs = node.get_lhs();

    /// we have to only solve ODEs under original derivative block where lhs is variable
    if (!derivative_block || !differential_equation || !lhs->is_var_name()) {
        return;
    }

    auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();

    if (name->is_prime_name()) {
        auto varname = "D" + name->get_node_name();
        node.set_lhs(std::make_shared<ast::Name>(new ast::String(varname)));
        if (program_symtab->lookup(varname) == nullptr) {
            auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
            symbol->set_original_name(name->get_node_name());
            program_symtab->insert(symbol);
        }
        if (node_type == ast::AstNodeType::DERIVATIVE_ORIGINAL_JACOBIAN_BLOCK) {
            // rename any array variables (like s[0]) so SymPy can handle it
            auto rhs = node.get_rhs()->clone();
            auto variables_to_rename = collect_nodes(*rhs, {ast::AstNodeType::INDEXED_NAME});
            // map of old names to new names
            std::unordered_map<std::string, std::string> name_map;
            if (!variables_to_rename.empty()) {
                // TODO actually rename them
                for (const auto& var: variables_to_rename) {
                }
            }
            auto rhs_string = to_nmodl(node.get_rhs());
            auto diff2c = pywrap::EmbeddedPythonLoader::get_instance().api().diff2c;
            auto [jacobian, exception_message] = diff2c(rhs_string, name->get_node_name());
            if (!exception_message.empty()) {
                logger->warn("DerivativeOriginalVisitor :: python exception: {}",
                             exception_message);
            }
            // NOTE: LHS can be anything here, the equality is to keep `create_statement` from
            // complaining, we discard the LHS later
            // TODO SymPy cannot handle array indexing like s[0], so we should rename those
            auto statement = fmt::format("{} = {} / (1 - dt * ({}))", varname, varname, jacobian);
            auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                create_statement(statement));
            const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
                expr_statement->get_expression());
            node.set_rhs(std::shared_ptr<ast::Expression>(bin_expr->get_rhs()->clone()));
        }
    }
    // edge case: it's an array, not a scalar
    else if (name->is_indexed_name()) {
        // TODO
    }
}

void DerivativeOriginalVisitor::visit_program(ast::Program& node) {
    program_symtab = node.get_symbol_table();
    node.visit_children(*this);
    if (der_block_function) {
        auto der_node =
            new ast::DerivativeOriginalFunctionBlock(der_block_function->get_name(),
                                                     der_block_function->get_statement_block());
        node.emplace_back_node(der_node);
    }
    if (der_block_jacobian) {
        auto der_node =
            new ast::DerivativeOriginalJacobianBlock(der_block_jacobian->get_name(),
                                                     der_block_jacobian->get_statement_block());
        node.emplace_back_node(der_node);
    }

    // re-visit the AST since we now inserted the DERIVATIVE_ORIGINAL block
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

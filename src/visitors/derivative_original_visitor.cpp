/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/derivative_original_visitor.hpp"

#include "ast/all.hpp"
#include "lexer/token_mapping.hpp"
#include "pybind/pyembed.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"
#include <optional>
#include <utility>

namespace pywrap = nmodl::pybind_wrappers;

namespace nmodl {
namespace visitor {


void DerivativeOriginalVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    node.visit_children(*this);
    der_block_function = node.clone();
}


void DerivativeOriginalVisitor::visit_derivative_original_function_block(
    ast::DerivativeOriginalFunctionBlock& node) {
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

    if (name->is_prime_name() || name->is_indexed_name()) {
        std::string varname;
        if (name->is_prime_name()) {
            varname = "D" + name->get_node_name();
            logger->debug("DerivativeOriginalVisitor :: replacing {} with {} on LHS of {}",
                          name->get_node_name(),
                          varname,
                          to_nmodl(node));
            node.set_lhs(std::make_shared<ast::Name>(new ast::String(varname)));
            if (program_symtab->lookup(varname) == nullptr) {
                auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
                symbol->set_original_name(name->get_node_name());
                program_symtab->insert(symbol);
            }
        } else {
            varname = "D" + stringutils::remove_character(to_nmodl(node.get_lhs()), '\'');
            // we discard the RHS here so it can be anything (as long as NMODL considers it valid)
            auto statement = fmt::format("{} = {}", varname, varname);
            logger->debug("DerivativeOriginalVisitor :: replacing {} with {} on LHS of {}",
                          to_nmodl(node.get_lhs()),
                          varname,
                          to_nmodl(node));
            auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                create_statement(statement));
            const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
                expr_statement->get_expression());
            node.set_lhs(std::shared_ptr<ast::Expression>(bin_expr->get_lhs()->clone()));
            // TODO add symbol?
        }
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

    // re-visit the AST since we now inserted the DERIVATIVE_ORIGINAL block
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

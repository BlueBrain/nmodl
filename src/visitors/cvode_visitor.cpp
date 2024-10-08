/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/cvode_visitor.hpp"

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

static int get_index(const ast::IndexedName& node) {
    return std::stoi(to_nmodl(node.get_length()));
}

void CvodeVisitor::visit_conserve(ast::Conserve& node) {
    logger->debug("CvodeVisitor :: CONSERVE statement: {}", to_nmodl(node));
    std::string conserve_equation_statevar;
    if (node.get_react()->is_react_var_name()) {
        conserve_equation_statevar = node.get_react()->get_node_name();
    }
    auto conserve_equation_str = to_nmodl(*node.get_expr());
    logger->debug("CvodeVisitor :: --> replace ODE for state var {} with equation {}",
                  conserve_equation_statevar,
                  conserve_equation_str);
    conserve_equations[conserve_equation_statevar] = conserve_equation_str;
}


static auto get_name_map(const ast::Expression& node, const std::string& name) {
    std::unordered_map<std::string, int> name_map;
    // all of the "reserved" symbols
    auto reserved_symbols = get_external_functions();
    // all indexed vars
    auto indexed_vars = collect_nodes(node, {ast::AstNodeType::INDEXED_NAME});
    for (const auto& var: indexed_vars) {
        if (!name_map.count(var->get_node_name()) && var->get_node_name() != name &&
            std::none_of(reserved_symbols.begin(), reserved_symbols.end(), [&var](const auto item) {
                return var->get_node_name() == item;
            })) {
            logger->debug(
                "CvodeVisitor :: adding INDEXED_VARIABLE {} to "
                "node_map",
                var->get_node_name());
            name_map[var->get_node_name()] = get_index(
                *std::dynamic_pointer_cast<const ast::IndexedName>(var));
        }
    }
    return name_map;
}

void CvodeVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    node.visit_children(*this);
    derivative_block = std::shared_ptr<ast::DerivativeBlock>(node.clone());
}


void CvodeVisitor::visit_cvode_block(ast::CvodeBlock& node) {
    in_cvode_block = true;
    node.visit_children(*this);
    in_cvode_block = false;
}

void CvodeVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    in_differential_equation = true;
    node.visit_children(*this);
    in_differential_equation = false;
}


void CvodeVisitor::visit_statement_block(ast::StatementBlock& node) {
    node.visit_children(*this);
    if (in_cvode_block) {
        ++block_index;
    }
}


void CvodeVisitor::visit_binary_expression(ast::BinaryExpression& node) {
    const auto& lhs = node.get_lhs();

    /// we have to only solve ODEs under original derivative block where lhs is variable
    if (!in_cvode_block || !in_differential_equation || !lhs->is_var_name()) {
        return;
    }

    auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();

    if (name->is_prime_name()) {
        auto varname = "D" + name->get_node_name();
        logger->debug("CvodeVisitor :: replacing {} with {} on LHS of {}",
                      name->get_node_name(),
                      varname,
                      to_nmodl(node));
        node.set_lhs(std::make_shared<ast::Name>(new ast::String(varname)));
        if (program_symtab->lookup(varname) == nullptr) {
            auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
            symbol->set_original_name(name->get_node_name());
            program_symtab->insert(symbol);
        }
        // case: there is a variable being CONSERVEd, but it's not the current one
        if (!conserve_equations.empty() && !conserve_equations.count(name->get_node_name())) {
            auto rhs = node.get_rhs();
            auto nodes = collect_nodes(*node.get_rhs(), {ast::AstNodeType::VAR_NAME});
            for (auto& n: nodes) {
                if (conserve_equations.count(n->get_node_name())) {
                    auto statement = fmt::format("{} = {}", n->get_node_name(), conserve_equations[n->get_node_name()]);
                    logger->debug("CvodeVisitor :: replacing CONSERVEd variable {} with {} in {}",
                                  n->get_node_name(),
                                  conserve_equations[n->get_node_name()],
                                  to_nmodl(*node.get_rhs()));
                    auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                        create_statement(statement));
                    const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
                        expr_statement->get_expression());
                    auto thing = std::shared_ptr<ast::Expression>(bin_expr->get_rhs()->clone());
                    n = std::move(std::dynamic_pointer_cast<ast::Expression>(thing));
                    std::cout << to_nmodl(*n) << std::endl;
                }
            }
        }
        std::cout << to_nmodl(node) << std::endl;
        if (block_index == BlockIndex::JACOBIAN) {
            auto rhs = node.get_rhs();
            // map of all indexed symbols (need special treatment in SymPy)
            auto name_map = get_name_map(*rhs, name->get_node_name());
            auto diff2c = pywrap::EmbeddedPythonLoader::get_instance().api().diff2c;
            auto [jacobian,
                  exception_message] = diff2c(to_nmodl(*rhs), name->get_node_name(), name_map);
            if (!exception_message.empty()) {
                logger->warn("CvodeVisitor :: python exception: {}", exception_message);
            }
            // NOTE: LHS can be anything here, the equality is to keep `create_statement` from
            // complaining, we discard the LHS later
            auto statement = fmt::format("{} = {} / (1 - dt * ({}))", varname, varname, jacobian);
            logger->debug("CvodeVisitor :: replacing statement {} with {}",
                          to_nmodl(node),
                          statement);
            auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                create_statement(statement));
            const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
                expr_statement->get_expression());
            node.set_rhs(std::shared_ptr<ast::Expression>(bin_expr->get_rhs()->clone()));
        }
    }
}

void CvodeVisitor::visit_program(ast::Program& node) {
    program_symtab = node.get_symbol_table();
    node.visit_children(*this);
    if (derivative_block) {
        auto der_node = new ast::CvodeBlock(derivative_block->get_name(),
                                            derivative_block->get_statement_block(),
                                            std::shared_ptr<ast::StatementBlock>(
                                                derivative_block->get_statement_block()->clone()));
        node.emplace_back_node(der_node);
    }

    for (const auto& [key, value]: conserve_equations) {
        std::cout << key << ", " << value << std::endl;
    }

    // re-visit the AST since we now inserted the CVODE block
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

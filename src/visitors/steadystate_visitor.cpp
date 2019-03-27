/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>

#include "codegen/codegen_naming.hpp"
#include "parser/nmodl_driver.hpp"
#include "steadystate_visitor.hpp"
#include "symtab/symbol.hpp"
#include "utils/logger.hpp"
#include "utils/string_utils.hpp"
#include "visitor_utils.hpp"
#include "visitors/lookup_visitor.hpp"

namespace nmodl {

using symtab::syminfo::NmodlType;

std::shared_ptr<ast::Statement> create_lineq(const std::string& eq) {
    nmodl::parser::NmodlDriver driver;
    auto nmodl_text = "LINEAR dummy { " + eq + " }";
    driver.parse_string(nmodl_text);
    auto ast = driver.ast();
    auto linblock = std::dynamic_pointer_cast<ast::LinearBlock>(ast->blocks[0]);
    auto statement = std::shared_ptr<ast::Statement>(
        linblock->get_statement_block()->get_statements()[0]->clone());
    return statement;
}

std::shared_ptr<ast::Statement> create_nonlineq(const std::string& eq) {
    nmodl::parser::NmodlDriver driver;
    auto nmodl_text = "NONLINEAR dummy { " + eq + " }";
    driver.parse_string(nmodl_text);
    auto ast = driver.ast();
    auto nonlinblock = std::dynamic_pointer_cast<ast::NonLinearBlock>(ast->blocks[0]);
    auto statement = std::shared_ptr<ast::Statement>(
        nonlinblock->get_statement_block()->get_statements()[0]->clone());
    return statement;
}

void SteadystateVisitor::visit_diff_eq_expression(ast::DiffEqExpression* node) {
    if (is_ss_deriv_block) {
        auto& lhs = node->get_expression()->lhs;
        auto& rhs = node->get_expression()->rhs;
        if (!lhs->is_var_name()) {
            logger->warn(
                "SteadystateVisitor :: LHS of differential equation is not a VariableName");
            return;
        }
        auto lhs_name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();
        if ((lhs_name->is_indexed_name() &&
             !std::dynamic_pointer_cast<ast::IndexedName>(lhs_name)->get_name()->is_prime_name()) ||
            (!lhs_name->is_indexed_name() && !lhs_name->is_prime_name())) {
            logger->warn("SteadystateVisitor :: LHS of differential equation is not a PrimeName");
            return;
        }
        // replace "x' = ..." with "~ ... = 0"
        std::string new_eq = "~ " + to_nmodl(rhs.get()) + " = 0";
        new_eqs.push_back(new_eq);
        statements_to_remove.insert(current_expression_statement);
    }
}

void SteadystateVisitor::visit_expression_statement(ast::ExpressionStatement* node) {
    auto prev_expression_statement = current_expression_statement;
    current_expression_statement = node;
    node->visit_children(this);
    current_expression_statement = prev_expression_statement;
}

void SteadystateVisitor::visit_statement_block(ast::StatementBlock* node) {
    auto prev_statement_block = current_statement_block;
    current_statement_block = node;
    node->visit_children(this);
    // remove processed statements from current statement block
    remove_statements_from_block(current_statement_block, statements_to_remove);
    current_statement_block = prev_statement_block;
}

void SteadystateVisitor::visit_derivative_block(ast::DerivativeBlock* node) {
    new_eqs.clear();
    statements_to_remove.clear();
    current_statement_block = nullptr;
    current_expression_statement = nullptr;

    bool is_ss_linear_block = (ss_linear_blocks.find(node) != ss_linear_blocks.cend());
    bool is_ss_nonlinear_block = (ss_nonlinear_blocks.find(node) != ss_nonlinear_blocks.cend());

    is_ss_deriv_block = is_ss_linear_block || is_ss_nonlinear_block;

    node->visit_children(this);

    auto ss_deriv_block = node->get_statement_block();
    // remove any remaining statements to be replaced
    remove_statements_from_block(ss_deriv_block.get(), statements_to_remove);
    // add new statements
    for (const auto& eq: new_eqs) {
        logger->debug("SteadystateVisitor :: -> adding statement: {}", eq);
        if (is_ss_linear_block) {
            ss_deriv_block->addStatement(create_lineq(eq));
        } else {
            ss_deriv_block->addStatement(create_nonlineq(eq));
        }
    }
}

void SteadystateVisitor::visit_program(ast::Program* node) {
    ss_linear_blocks.clear();
    ss_nonlinear_blocks.clear();
    current_statement_block = nullptr;
    current_expression_statement = nullptr;

    // get DERIVATIVE blocks
    const auto& deriv_blocks = AstLookupVisitor().lookup(node, ast::AstNodeType::DERIVATIVE_BLOCK);

    // get list of STEADYSTATE solve statements with names & methods
    const auto& solve_block_nodes = AstLookupVisitor().lookup(node, ast::AstNodeType::SOLVE_BLOCK);

    // clone DERIVATE blocks that have STEADYSTATE solves
    for (const auto& solve_block_ptr: solve_block_nodes) {
        if (auto solve_block = std::dynamic_pointer_cast<ast::SolveBlock>(solve_block_ptr)) {
            if (solve_block->get_steadystate()) {
                // for each STEADYSTATE statement, get method & derivative block
                std::string solve_block_name = solve_block->get_block_name()->get_value()->eval();
                const auto& steadystate_method =
                    solve_block->get_steadystate()->get_value()->eval();
                logger->debug(
                    "SteadystateVisitor :: Found STEADYSTATE SOLVE statement: using {} for {}",
                    steadystate_method, solve_block_name);
                ast::DerivativeBlock* deriv_block_ptr = nullptr;
                for (const auto& block_ptr: deriv_blocks) {
                    auto deriv_block = std::dynamic_pointer_cast<ast::DerivativeBlock>(block_ptr);
                    if (deriv_block->get_node_name() == solve_block_name) {
                        logger->debug(
                            "SteadystateVisitor :: -> found corresponding DERIVATIVE block: {}",
                            solve_block_name);
                        deriv_block_ptr = deriv_block.get();
                    }
                }
                if (deriv_block_ptr != nullptr) {
                    // make a clone of derivative block with "_steadystate" suffix
                    auto ss_block = std::shared_ptr<ast::DerivativeBlock>(deriv_block_ptr->clone());
                    auto ss_name = ss_block->get_name();
                    ss_name->set_name(ss_name->get_value()->get_value() + "_steadystate");
                    auto ss_name_clone = std::shared_ptr<ast::Name>(ss_name->clone());
                    ss_block->set_name(std::move(ss_name));
                    logger->debug("SteadystateVisitor :: Adding new DERIVATIVE block: {}",
                                  ss_block->get_node_name());
                    node->addNode(ss_block);
                    // add it to the set of either linear or non-linear steadystate blocks
                    if (steadystate_method == codegen::naming::SPARSE_METHOD) {
                        ss_linear_blocks.insert(ss_block.get());
                    } else if (steadystate_method == codegen::naming::DERIVIMPLICIT_METHOD) {
                        ss_nonlinear_blocks.insert(ss_block.get());
                    } else {
                        logger->warn(
                            "SteadystateVisitor :: solve method {} not supported for "
                            "STEADYSTATE",
                            steadystate_method);
                    }
                    // update SOLVE statement:
                    // set name to point to new (NON)LINEAR block
                    solve_block->set_block_name(std::move(ss_name_clone));
                    // set STEADYSTATE to nullptr for (NON)LINEAR block
                    solve_block->set_steadystate(nullptr);
                } else {
                    logger->warn(
                        "SteadystateVisitor :: Could not find derivative block {} for "
                        "STEADYSTATE",
                        solve_block_name);
                }
            }
        }
    }

    // replace each DiffEq
    // x' = ...
    // in the new steadystate derivative blocks with
    // the corresponding (NON)LINEAR steady state equation
    // ~ ... = 0
    node->visit_children(this);

    // change the steady state DERIVATIVE blocks to (NON)LINEAR blocks
    auto blocks = node->get_blocks();
    for (auto* ss_linear_block: ss_linear_blocks) {
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->get() == ss_linear_block) {
                logger->debug("SteadystateVisitor :: changing block {} from DERIVATIVE to LINEAR",
                              ss_linear_block->get_node_name());
                auto linblock = std::make_shared<ast::LinearBlock>(
                    std::move(ss_linear_block->get_name()), ast::NameVector{},
                    std::move(ss_linear_block->get_statement_block()));
                ModToken tok{};
                linblock->set_token(tok);
                *it = linblock;
            }
        }
    }
    for (auto* ss_nonlinear_block: ss_nonlinear_blocks) {
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->get() == ss_nonlinear_block) {
                logger->debug(
                    "SteadystateVisitor :: changing block {} from DERIVATIVE to NONLINEAR",
                    ss_nonlinear_block->get_node_name());
                auto nonlinblock = std::make_shared<ast::NonLinearBlock>(
                    std::move(ss_nonlinear_block->get_name()), ast::NameVector{},
                    std::move(ss_nonlinear_block->get_statement_block()));
                ModToken tok{};
                nonlinblock->set_token(tok);
                *it = nonlinblock;
            }
        }
    }
    node->set_blocks(std::move(blocks));
}

}  // namespace nmodl

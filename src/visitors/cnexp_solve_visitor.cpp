/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <sstream>

#include "codegen/codegen_naming.hpp"
#include "parser/diffeq_driver.hpp"
#include "symtab/symbol.hpp"
#include "utils/logger.hpp"
#include "visitors/cnexp_solve_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {

void CnexpSolveVisitor::visit_solve_block(ast::SolveBlock* node) {
    auto name = node->get_block_name()->get_node_name();
    auto method = node->get_method();
    auto solve_method = method ? method->get_value()->eval() : "";
    solve_blocks[name] = solve_method;
}

void CnexpSolveVisitor::visit_derivative_block(ast::DerivativeBlock* node) {
    derivative_block_name = node->get_name()->get_node_name();
    node->visit_children(this);
}

void CnexpSolveVisitor::visit_diff_eq_expression(ast::DiffEqExpression* node) {
    differential_equation = true;
    node->visit_children(this);
    differential_equation = false;
}

void CnexpSolveVisitor::visit_binary_expression(ast::BinaryExpression* node) {
    auto& lhs = node->lhs;
    auto& rhs = node->rhs;
    auto& op = node->op;

    /// we have to only solve binary expressions in derivative block
    if (!differential_equation) {
        return;
    }

    /// lhs of the expression should be variable
    if (!lhs->is_var_name()) {
        logger->warn("LHS of differential equation is not a VariableName");
        return;
    }

    auto solve_method = solve_blocks[derivative_block_name];
    auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();

    if (name->is_prime_name()) {
        auto equation = nmodl::to_nmodl(node);
        parser::DiffeqDriver diffeq_driver;

        if (solve_method == cnexp_method) {
            std::string solution;
            /// check if ode can be solved with cnexp method
            if (diffeq_driver.cnexp_possible(equation, solution)) {
                auto statement = create_statement(solution);
                auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                    statement);
                auto bin_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(
                    expr_statement->get_expression());
                lhs.reset(bin_expr->lhs->clone());
                rhs.reset(bin_expr->rhs->clone());
                keep_derivative_block[derivative_block_name] = false;
            } else {
                logger->error("cnexp solver not possible");
                keep_derivative_block[derivative_block_name] = true;
            }
        } else if (solve_method == euler_method) {
            std::string solution = diffeq_driver.solve(equation, solve_method);
            auto statement = create_statement(solution);
            auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(statement);
            auto bin_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(
                expr_statement->get_expression());
            lhs.reset(bin_expr->lhs->clone());
            rhs.reset(bin_expr->rhs->clone());
            keep_derivative_block[derivative_block_name] = false;
        } else if (solve_method == derivimplicit_method) {
            auto varname = "D" + name->get_node_name();
            auto variable = new ast::Name(new ast::String(varname));
            lhs.reset(variable);
            if (program_symtab->lookup(varname) == nullptr) {
                auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
                symbol->set_original_name(name->get_node_name());
                symbol->created_from_state();
                program_symtab->insert(symbol);
            }
            keep_derivative_block[derivative_block_name] = true;
        } else {
            logger->error("solver method '{}' not supported", solve_method);
        }
    }
}

void CnexpSolveVisitor::visit_program(ast::Program* node) {
    program_symtab = node->get_symbol_table();
    node->visit_children(this);

    // After having attempted to solve all derivative blocks we check for the ones that can
    // now be removed
    auto program_blocks = node->get_blocks();
    bool update = false;
    for(auto block_it = program_blocks.begin(); block_it < program_blocks.end(); block_it++) {
        if ((*block_it)->is_derivative_block()) {
            auto deriv_block = std::dynamic_pointer_cast<ast::DerivativeBlock>(*block_it);
            auto deriv_block_name = deriv_block->get_name()->get_node_name();
            if((keep_derivative_block.find(deriv_block_name) != keep_derivative_block.cend()) and
               !keep_derivative_block[deriv_block_name]) {
                program_blocks.erase(block_it);
                update = true;
            }
        }
    }
    if (update) {
        node->set_blocks(std::move(program_blocks));
    }
}

}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>

#include "codegen/codegen_naming.hpp"
#include "symtab/symbol.hpp"
#include "utils/logger.hpp"
#include "visitor_utils.hpp"
#include "visitors/sympy_solver_visitor.hpp"


namespace py = pybind11;
using namespace py::literals;

namespace nmodl {

using symtab::syminfo::NmodlType;

std::shared_ptr<ast::FunctorBlock> SympySolverVisitor::construct_functor_block(
    const std::string& name,
    const std::vector<std::string>& statements) {
    ast::StatementVector statement_vector;
    for (const auto& statement: statements) {
        statement_vector.push_back(create_statement(statement));
    }
    auto statement_block = std::make_shared<ast::StatementBlock>(statement_vector);
    auto name_str = std::make_shared<ast::String>(name);
    auto ast_name = std::make_shared<ast::Name>(name_str);
    return std::make_shared<ast::FunctorBlock>(ast_name, statement_block);
}

void SympySolverVisitor::replace_binary_expression(ast::BinaryExpression* bin_expr,
                                                   const std::string& new_binary_expr) {
    auto& lhs = bin_expr->lhs;
    auto& rhs = bin_expr->rhs;
    auto new_statement = create_statement(new_binary_expr);
    auto new_expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(new_statement);
    auto new_bin_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(
        new_expr_statement->get_expression());
    lhs.reset(new_bin_expr->lhs->clone());
    rhs.reset(new_bin_expr->rhs->clone());
}

void SympySolverVisitor::visit_diff_eq_expression(ast::DiffEqExpression* node) {
    auto& lhs = node->get_expression()->lhs;
    auto& rhs = node->get_expression()->rhs;

    if (!lhs->is_var_name()) {
        logger->warn("SympySolverVisitor :: LHS of differential equation is not a VariableName");
        return;
    }
    auto lhs_name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();
    if (!lhs_name->is_prime_name()) {
        logger->warn("SympySolverVisitor :: LHS of differential equation is not a PrimeName");
        return;
    }
    const auto node_as_nmodl = to_nmodl_for_sympy(node);
    const auto locals = py::dict("equation_string"_a = node_as_nmodl,
                                 "t_var"_a = codegen::naming::NTHREAD_T_VARIABLE,
                                 "dt_var"_a = codegen::naming::NTHREAD_DT_VARIABLE, "vars"_a = vars,
                                 "use_pade_approx"_a = use_pade_approx);
    if (solve_method == euler_method) {
        logger->debug("SympySolverVisitor :: EULER - solving: {}", node_as_nmodl);
        // replace x' = f(x) differential equation
        // with forwards Euler timestep:
        // x = x + f(x) * dt
        py::exec(R"(
                from nmodl.ode import forwards_euler2c
                exception_message = ""
                try:
                    solution = forwards_euler2c(equation_string, dt_var, vars)
                except Exception as e:
                    # if we fail, fail silently and return empty string
                    solution = ""
                    exception_message = str(e)
            )",
                 py::globals(), locals);
    } else if (solve_method == cnexp_method) {
        // replace x' = f(x) differential equation
        // with analytic solution for x(t+dt) in terms of x(t)
        // x = ...
        logger->debug("SympySolverVisitor :: CNEXP - solving: {}", node_as_nmodl);
        py::exec(R"(
                from nmodl.ode import integrate2c
                exception_message = ""
                try:
                    solution = integrate2c(equation_string, t_var, dt_var, vars, use_pade_approx)
                except Exception as e:
                    # if we fail, fail silently and return empty string
                    solution = ""
                    exception_message = str(e)
            )",
                 py::globals(), locals);
    } else {
        // for other solver methods: just collect the ODEs & return
        logger->debug("SympySolverVisitor :: adding ODE system: {}", to_nmodl_for_sympy(node));
        ode_system.push_back(to_nmodl_for_sympy(node));
        binary_expressions_to_replace.push_back(node->get_expression());
        return;
    }
    // replace ODE with solution in AST
    auto solution = locals["solution"].cast<std::string>();
    logger->debug("SympySolverVisitor :: -> solution: {}", solution);

    auto exception_message = locals["exception_message"].cast<std::string>();
    if (!exception_message.empty()) {
        logger->warn("SympySolverVisitor :: python exception: " + exception_message);
        return;
    }

    if (!solution.empty()) {
        replace_binary_expression(node->get_expression().get(), solution);
    } else {
        logger->warn("SympySolverVisitor :: solution to differential equation not possible");
    }
}

void SympySolverVisitor::visit_derivative_block(ast::DerivativeBlock* node) {
    // get all vars for this block, i.e. global vars + local vars
    vars = global_vars;
    if (auto symtab = node->get_statement_block()->get_symbol_table()) {
        auto localvars = symtab->get_variables_with_properties(NmodlType::local_var);
        for (const auto& v: localvars) {
            vars.insert(v->get_name());
        }
    }

    // get user specified solve method for this block
    solve_method = derivative_block_solve_method[node->get_node_name()];

    // visit each differential equation:
    // -for CNEXP or EULER, each equation is independent & is replaced with its solution
    // -otherwise, each equation is added to ode_system (and to binary_expressions_to_replace)
    ode_system.clear();
    binary_expressions_to_replace.clear();
    node->visit_children(this);

    // solve system of ODEs in ode_system
    if (!ode_system.empty()) {
        logger->debug("SympySolverVisitor :: Solving {} system of ODEs", solve_method);
        auto locals = py::dict("equation_strings"_a = ode_system,
                               "t_var"_a = codegen::naming::NTHREAD_T_VARIABLE,
                               "dt_var"_a = codegen::naming::NTHREAD_DT_VARIABLE, "vars"_a = vars,
                               "do_cse"_a = elimination);
        py::exec(R"(
                from nmodl.ode import solve_ode_system
                exception_message = ""
                try:
                    solutions, new_local_vars = solve_ode_system(equation_strings, t_var, dt_var, vars, do_cse)
                except Exception as e:
                    # if we fail, fail silently and return empty string
                    solutions = [""]
                    new_local_vars = [""]
                    exception_message = str(e)
            )",
                 py::globals(), locals);
        // returns a vector of solutions, i.e. new statements to add to block:
        auto solutions = locals["solutions"].cast<std::vector<std::string>>();
        // and a vector of new local variables that need to be declared in the block:
        auto new_local_vars = locals["new_local_vars"].cast<std::vector<std::string>>();
        auto exception_message = locals["exception_message"].cast<std::string>();
        if (!exception_message.empty()) {
            logger->warn("SympySolverVisitor :: solve_ode_system python exception: " +
                         exception_message);
            return;
        }
        // sanity check: must have at least as many solutions as ODE's to replace:
        if (solutions.size() < binary_expressions_to_replace.size()) {
            logger->warn("SympySolverVisitor :: Solve failed: fewer solutions than ODE's");
            return;
        }
        // declare new local vars
        if (!new_local_vars.empty()) {
            for (const auto& new_local_var: new_local_vars) {
                logger->debug("SympySolverVisitor :: -> declaring new local variable: {}",
                              new_local_var);
                add_local_variable(node->get_statement_block().get(), new_local_var);
            }
        }
        if (solve_method == "sparse") {
            // add new statements: firstly by replacing old ODE binary expressions
            auto sol = solutions.cbegin();
            for (auto binary_expr: binary_expressions_to_replace) {
                logger->debug("SympySolverVisitor :: -> replacing {} with statement: {}",
                              to_nmodl_for_sympy(binary_expr.get()), *sol);
                replace_binary_expression(binary_expr.get(), *sol);
                ++sol;
            }
            // then by adding the rest as new statements to the block
            // get a copy of existing statements in block
            auto statements = node->get_statement_block()->get_statements();
            while (sol != solutions.cend()) {
                // add new statements to block
                logger->debug("SympySolverVisitor :: -> adding statement: {}", *sol);
                statements.push_back(create_statement(*sol));
                ++sol;
            }
            // replace old set of statements in AST with new one
            node->get_statement_block()->set_statements(std::move(statements));
        } else if (solve_method == "derivimplicit") {
            // replace old ODE binary expressions
            auto sol = solutions.cbegin();
            for (auto binary_expr: binary_expressions_to_replace) {
                logger->debug("SympySolverVisitor :: -> replacing {} with statement: {}",
                              to_nmodl_for_sympy(binary_expr.get()), *sol);
                replace_binary_expression(binary_expr.get(), *sol);
                ++sol;
            }
            // put F, J into new functor with same name as this block
            auto functor_eqs = std::vector<std::string>{sol, solutions.cend()};
            new_functor_blocks.push_back(
                construct_functor_block(node->get_node_name(), functor_eqs));
        }
    }
}

void SympySolverVisitor::visit_program(ast::Program* node) {
    global_vars = get_global_vars(node);

    // get list of solve statements with names & methods
    AstLookupVisitor ast_lookup_visitor;
    auto solve_block_nodes = ast_lookup_visitor.lookup(node, ast::AstNodeType::SOLVE_BLOCK);
    for (const auto& block: solve_block_nodes) {
        if (auto block_ptr = std::dynamic_pointer_cast<ast::SolveBlock>(block)) {
            std::string solve_method = block_ptr->get_method()->get_value()->eval();
            std::string block_name = block_ptr->get_block_name()->get_value()->eval();
            logger->debug("SympySolverVisitor :: Found SOLVE statement: using {} for {}",
                          solve_method, block_name);
            derivative_block_solve_method[block_name] = solve_method;
            // temporary hack to avoid current derivimplicit codegen:
            if (solve_method == "derivimplicit") {
                auto s = std::make_shared<ast::String>("euler");
                block_ptr->get_method()->set_value(std::move(s));
            }
        }
    }

    node->visit_children(this);

    for (auto functor_block: new_functor_blocks) {
        node->addNode(functor_block);
        std::cout << to_nmodl(functor_block.get()) << std::endl;
    }
}

}  // namespace nmodl
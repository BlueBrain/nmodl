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
#include "utils/string_utils.hpp"
#include "visitors/lookup_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace nmodl {

using symtab::syminfo::NmodlType;

void SympySolverVisitor::init_block_data(ast::Node* node) {
    // clear any previous data
    expression_statements.clear();
    eq_system.clear();
    last_expression_statement = nullptr;
    block_with_expression_statements = nullptr;
    eq_system_is_valid = true;
    // get set of local block vars & global vars
    vars = global_vars;
    if (auto symtab = node->get_statement_block()->get_symbol_table()) {
        auto localvars = symtab->get_variables_with_properties(NmodlType::local_var);
        for (const auto& localvar: localvars) {
            vars.insert(localvar->get_name());
        }
    }
}

void SympySolverVisitor::replace_diffeq_expression(ast::DiffEqExpression* expr,
                                                   const std::string& new_expr) {
    auto new_statement = create_statement(new_expr);
    auto new_expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(new_statement);
    auto new_bin_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(
        new_expr_statement->get_expression());
    expr->set_expression(std::move(new_bin_expr));
}

void SympySolverVisitor::check_expr_statements_in_same_block() {
    /// all ode/kinetic/(non)linear statements (typically) appear in the same statement block
    /// if this is not the case, for now return an error (and should instead use fallback solver)
    if (block_with_expression_statements != nullptr &&
        block_with_expression_statements != current_statement_block) {
        logger->warn(
            "SympySolverVisitor :: Coupled equations are appearing in different blocks - not "
            "supported");
        eq_system_is_valid = false;
    }
    block_with_expression_statements = current_statement_block;
}

ast::StatementVector::iterator SympySolverVisitor::get_solution_location_iterator(
    ast::StatementVector& statements) {
    // find out where to insert solutions in statement block
    // returns iterator pointing to the first element after the last (non)linear eq
    // so if there are no such elements, it returns statements.end()
    auto it = statements.begin();
    if (last_expression_statement != nullptr) {
        while ((it != statements.end()) &&
               (std::dynamic_pointer_cast<ast::ExpressionStatement>(*it).get() !=
                last_expression_statement)) {
            logger->debug("SympySolverVisitor :: {} != {}", to_nmodl((*it).get()),
                          to_nmodl(last_expression_statement));
            ++it;
        }
        if (it != statements.end()) {
            logger->debug("SympySolverVisitor :: {} == {}",
                          to_nmodl(std::dynamic_pointer_cast<ast::ExpressionStatement>(*it).get()),
                          to_nmodl(last_expression_statement));
            ++it;
        }
    }
    return it;
}

static bool has_local_statement(std::shared_ptr<ast::Statement> statement) {
    return !AstLookupVisitor().lookup(statement.get(), ast::AstNodeType::LOCAL_VAR).empty();
}

void SympySolverVisitor::construct_eigen_solver_block(
    const std::vector<std::string>& pre_solve_statements,
    const std::vector<std::string>& solutions,
    bool linear) {
    // find out where to insert solution in statement block
    auto& statements = block_with_expression_statements->statements;
    auto it = get_solution_location_iterator(statements);
    // insert pre-solve statements below last linear eq in block
    for (const auto& statement: pre_solve_statements) {
        logger->debug("SympySolverVisitor :: -> adding statement: {}", statement);
        it = statements.insert(it, create_statement(statement));
        ++it;
    }
    // make Eigen vector <-> state var assignments
    std::vector<std::string> setup_x_eqs;
    std::vector<std::string> update_state_eqs;
    for (int i = 0; i < state_vars.size(); i++) {
        auto statement = state_vars[i] + " = X[" + std::to_string(i) + "]";
        auto rev_statement = "X[" + std::to_string(i) + "] = " + state_vars[i];
        update_state_eqs.push_back(statement);
        setup_x_eqs.push_back(rev_statement);
        logger->debug("SympySolverVisitor :: setup_x: {}", rev_statement);
        logger->debug("SympySolverVisitor :: update_state: {}", statement);
    }
    // TODO: make unique name for Eigen vector if clashes
    if (vars.find("X") != vars.end()) {
        logger->error("SympySolverVisitor :: -> X conflicts with NMODL variable");
    }
    for (const auto& sol: solutions) {
        logger->debug("SympySolverVisitor :: -> adding statement: {}", sol);
    }
    // statements after last diff/linear/non-linear eq statement go into finalize_block
    ast::StatementVector finalize_statements{it, statements.end()};
    // remove them from the statement block
    statements.erase(it, statements.end());
    // also remove diff/linear/non-linear eq statements from the statement block
    remove_statements_from_block(block_with_expression_statements, expression_statements);
    // move any local variable declarations into variable_block
    ast::StatementVector variable_statements;
    // remaining statements in block should go into initialize_block
    ast::StatementVector initialize_statements;
    for (auto s: statements) {
        if (has_local_statement(s)) {
            variable_statements.push_back(s);
        } else {
            initialize_statements.push_back(s);
        }
    }
    // make statement blocks
    auto variable_block = std::make_shared<ast::StatementBlock>(std::move(variable_statements));
    auto initialize_block = std::make_shared<ast::StatementBlock>(std::move(initialize_statements));
    auto update_state_block = create_statement_block(update_state_eqs);
    auto finalize_block = std::make_shared<ast::StatementBlock>(std::move(finalize_statements));

    if (linear) {
        /// create eigen linear solver block
        setup_x_eqs.insert(setup_x_eqs.end(), solutions.begin(), solutions.end());
        auto setup_x_block = create_statement_block(setup_x_eqs);
        auto solver_block = std::make_shared<ast::EigenLinearSolverBlock>(
            variable_block, initialize_block, setup_x_block, update_state_block, finalize_block);
        /// replace statement block with solver block as it contains all statements
        ast::StatementVector solver_block_statements{
            std::make_shared<ast::ExpressionStatement>(solver_block)};
        block_with_expression_statements->set_statements(std::move(solver_block_statements));
    } else {
        /// create eigen newton solver block
        auto setup_x_block = create_statement_block(setup_x_eqs);
        auto functor_block = create_statement_block(solutions);
        auto solver_block =
            std::make_shared<ast::EigenNewtonSolverBlock>(variable_block, initialize_block,
                                                          setup_x_block, functor_block,
                                                          update_state_block, finalize_block);

        /// replace statement block with solver block as it contains all statements
        ast::StatementVector solver_block_statements{
            std::make_shared<ast::ExpressionStatement>(solver_block)};
        block_with_expression_statements->set_statements(std::move(solver_block_statements));
    }
}

void SympySolverVisitor::solve_linear_system(const std::vector<std::string>& pre_solve_statements) {
    // call sympy linear solver
    bool small_system = (eq_system.size() <= SMALL_LINEAR_SYSTEM_MAX_STATES);
    auto locals = py::dict("eq_strings"_a = eq_system, "state_vars"_a = state_vars, "vars"_a = vars,
                           "small_system"_a = small_system, "do_cse"_a = elimination);
    py::exec(R"(
                from nmodl.ode import solve_lin_system
                exception_message = ""
                try:
                    solutions, new_local_vars = solve_lin_system(eq_strings,
                                                                 state_vars,
                                                                 vars,
                                                                 small_system,
                                                                 do_cse)
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
    // may also return a python exception message:
    auto exception_message = locals["exception_message"].cast<std::string>();
    if (!exception_message.empty()) {
        logger->warn("SympySolverVisitor :: solve_lin_system python exception: " +
                     exception_message);
        return;
    }
    // find out where to insert solutions in statement block
    auto& statements = block_with_expression_statements->statements;
    auto it = get_solution_location_iterator(statements);
    if (small_system) {
        // for small number of state vars, linear solver
        // directly returns solution by solving symbolically at compile time
        logger->debug("SympySolverVisitor :: Solving *small* linear system of eqs");
        // declare new local vars
        if (!new_local_vars.empty()) {
            for (const auto& new_local_var: new_local_vars) {
                logger->debug("SympySolverVisitor :: -> declaring new local variable: {}",
                              new_local_var);
                add_local_variable(block_with_expression_statements, new_local_var);
            }
        }
        // insert pre-solve statements below last linear eq in block
        for (const auto& statement: pre_solve_statements) {
            logger->debug("SympySolverVisitor :: -> adding statement: {}", statement);
            it = statements.insert(it, create_statement(statement));
            ++it;
        }
        // then insert new solution statements
        for (const auto& sol: solutions) {
            logger->debug("SympySolverVisitor :: -> adding statement: {}", sol);
            it = statements.insert(it, create_statement(sol));
            ++it;
        }
        /// remove original lineq statements from the block
        remove_statements_from_block(block_with_expression_statements, expression_statements);
    } else {
        // otherwise it returns a linear matrix system to solve
        logger->debug("SympySolverVisitor :: Constructing linear newton solve block");
        construct_eigen_solver_block(pre_solve_statements, solutions, true);
    }
}

void SympySolverVisitor::solve_non_linear_system(
    const std::vector<std::string>& pre_solve_statements) {
    // call sympy non-linear solver
    auto locals = py::dict("equation_strings"_a = eq_system, "state_vars"_a = state_vars,
                           "vars"_a = vars);
    py::exec(R"(
                from nmodl.ode import solve_non_lin_system
                exception_message = ""
                try:
                    solutions = solve_non_lin_system(equation_strings,
                                                     state_vars,
                                                     vars)
                except Exception as e:
                    # if we fail, fail silently and return empty string
                    solutions = [""]
                    new_local_vars = [""]
                    exception_message = str(e)
                )",
             py::globals(), locals);
    // returns a vector of solutions, i.e. new statements to add to block:
    auto solutions = locals["solutions"].cast<std::vector<std::string>>();
    // may also return a python exception message:
    auto exception_message = locals["exception_message"].cast<std::string>();
    if (!exception_message.empty()) {
        logger->warn("SympySolverVisitor :: solve_non_lin_system python exception: " +
                     exception_message);
        return;
    }
    logger->debug("SympySolverVisitor :: Constructing eigen newton solve block");
    construct_eigen_solver_block(pre_solve_statements, solutions, false);
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

    check_expr_statements_in_same_block();

    const auto node_as_nmodl = to_nmodl_for_sympy(node);
    const auto locals = py::dict("equation_string"_a = node_as_nmodl,
                                 "t_var"_a = codegen::naming::NTHREAD_T_VARIABLE,
                                 "dt_var"_a = codegen::naming::NTHREAD_DT_VARIABLE, "vars"_a = vars,
                                 "use_pade_approx"_a = use_pade_approx);

    if (solve_method == codegen::naming::EULER_METHOD) {
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
    } else if (solve_method == codegen::naming::CNEXP_METHOD) {
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
        eq_system.push_back(to_nmodl_for_sympy(node));
        expression_statements.insert(current_expression_statement);
        last_expression_statement = current_expression_statement;
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
        replace_diffeq_expression(node, solution);
    } else {
        logger->warn("SympySolverVisitor :: solution to differential equation not possible");
    }
}

void SympySolverVisitor::visit_derivative_block(ast::DerivativeBlock* node) {
    /// clear information from previous block, get global vars + block local vars
    init_block_data(node);

    // get user specified solve method for this block
    solve_method = derivative_block_solve_method[node->get_node_name()];

    // visit each differential equation:
    //  - for CNEXP or EULER, each equation is independent & is replaced with its solution
    //  - otherwise, each equation is added to eq_system
    node->visit_children(this);

    if (eq_system_is_valid && !eq_system.empty()) {
        // solve system of ODEs in eq_system
        logger->debug("SympySolverVisitor :: Solving {} system of ODEs", solve_method);

        // construct implicit Euler equations from ODEs
        std::vector<std::string> pre_solve_statements;
        for (auto& eq: eq_system) {
            auto split_eq = stringutils::split_string(eq, '=');
            auto x_prime_split = stringutils::split_string(split_eq[0], '\'');
            auto x = stringutils::trim(x_prime_split[0]);
            auto dxdt = stringutils::trim(split_eq[1]);
            auto old_x = "old_" + x;  // TODO: do this properly, check name is unique
            // declare old_x
            logger->debug("SympySolverVisitor :: -> declaring new local variable: {}", old_x);
            add_local_variable(block_with_expression_statements, old_x);
            // assign old_x = x
            pre_solve_statements.push_back(old_x + " = " + x);
            eq = x + " = " + old_x + " + " + codegen::naming::NTHREAD_DT_VARIABLE + " * (" + dxdt +
                 ")";
            logger->debug("SympySolverVisitor :: -> constructed euler eq: {}", eq);
        }

        if (solve_method == codegen::naming::SPARSE_METHOD) {
            solve_linear_system(pre_solve_statements);
        } else if (solve_method == codegen::naming::DERIVIMPLICIT_METHOD) {
            solve_non_linear_system(pre_solve_statements);
        } else {
            logger->error("SympySolverVisitor :: Solve method {} not supported", solve_method);
        }
    }
}

void SympySolverVisitor::visit_lin_equation(ast::LinEquation* node) {
    check_expr_statements_in_same_block();
    std::string lin_eq = to_nmodl_for_sympy(node->get_left_linxpression().get());
    lin_eq += " = ";
    lin_eq += to_nmodl_for_sympy(node->get_linxpression().get());
    eq_system.push_back(lin_eq);
    expression_statements.insert(current_expression_statement);
    last_expression_statement = current_expression_statement;
    logger->debug("SympySolverVisitor :: adding linear eq: {}", lin_eq);
}

void SympySolverVisitor::visit_linear_block(ast::LinearBlock* node) {
    logger->debug("SympySolverVisitor :: found LINEAR block: {}", node->get_node_name());

    /// clear information from previous block, get global vars + block local vars
    init_block_data(node);

    // collect linear equations
    node->visit_children(this);

    if (eq_system_is_valid && !eq_system.empty()) {
        solve_linear_system();
    }
}

void SympySolverVisitor::visit_non_lin_equation(ast::NonLinEquation* node) {
    check_expr_statements_in_same_block();
    std::string non_lin_eq = to_nmodl_for_sympy(node->get_lhs().get());
    non_lin_eq += " = ";
    non_lin_eq += to_nmodl_for_sympy(node->get_rhs().get());
    eq_system.push_back(non_lin_eq);
    expression_statements.insert(current_expression_statement);
    last_expression_statement = current_expression_statement;
    logger->debug("SympySolverVisitor :: adding non-linear eq: {}", non_lin_eq);
}

void SympySolverVisitor::visit_non_linear_block(ast::NonLinearBlock* node) {
    logger->debug("SympySolverVisitor :: found NONLINEAR block: {}", node->get_node_name());

    /// clear information from previous block, get global vars + block local vars
    init_block_data(node);

    // collect non-linear equations
    node->visit_children(this);

    if (eq_system_is_valid && !eq_system.empty()) {
        solve_non_linear_system();
    }
}

void SympySolverVisitor::visit_expression_statement(ast::ExpressionStatement* node) {
    auto prev_expression_statement = current_expression_statement;
    current_expression_statement = node;
    node->visit_children(this);
    current_expression_statement = prev_expression_statement;
}

void SympySolverVisitor::visit_statement_block(ast::StatementBlock* node) {
    auto prev_statement_block = current_statement_block;
    current_statement_block = node;
    node->visit_children(this);
    current_statement_block = prev_statement_block;
}

void SympySolverVisitor::visit_program(ast::Program* node) {
    global_vars = get_global_vars(node);

    // get list of solve statements with names & methods
    AstLookupVisitor ast_lookup_visitor;
    auto solve_block_nodes = ast_lookup_visitor.lookup(node, ast::AstNodeType::SOLVE_BLOCK);
    for (const auto& block: solve_block_nodes) {
        if (auto block_ptr = std::dynamic_pointer_cast<ast::SolveBlock>(block)) {
            std::string solve_method;
            if (block_ptr->get_method()) {
                // Note: solve method name is an optional parameter
                // LINEAR and NONLINEAR blocks do not have solve method specified
                solve_method = block_ptr->get_method()->get_value()->eval();
            }
            std::string block_name = block_ptr->get_block_name()->get_value()->eval();
            logger->debug("SympySolverVisitor :: Found SOLVE statement: using {} for {}",
                          solve_method, block_name);
            derivative_block_solve_method[block_name] = solve_method;
        }
    }

    // get list of state vars
    state_vars.clear();
    if (auto symtab = node->get_symbol_table()) {
        auto statevars = symtab->get_variables_with_properties(NmodlType::state_var);
        for (const auto& v: statevars) {
            const auto& varname = v->get_name();
            state_vars.push_back(varname);
        }
    }

    node->visit_children(this);
}

}  // namespace nmodl

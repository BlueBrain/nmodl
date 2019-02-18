#include "visitors/sympy_solver_visitor.hpp"
#include "codegen/codegen_naming.hpp"
#include "symtab/symbol.hpp"
#include "utils/logger.hpp"
#include "visitor_utils.hpp"
#include <iostream>
using namespace ast;
namespace py = pybind11;
using namespace py::literals;
using namespace syminfo;

void SympySolverVisitor::visit_solve_block(SolveBlock* node) {
    auto method = node->get_method();
    if (method) {
        solve_method = method->get_value()->eval();
    }
}

void SympySolverVisitor::visit_diff_eq_expression(DiffEqExpression* node) {
    if (solve_method != cnexp_method) {
        logger->warn(
            "SympySolverVisitor: solve method not cnexp, so not integrating "
            "expression analytically");
        return;
    }

    auto& lhs = node->get_expression()->lhs;
    auto& rhs = node->get_expression()->rhs;

    if (!lhs->is_var_name()) {
        logger->warn("SympySolverVisitor: LHS of differential equation is not a VariableName");
        return;
    }
    auto lhs_name = std::dynamic_pointer_cast<VarName>(lhs)->get_name();
    if (!lhs_name->is_prime_name()) {
        logger->warn("SympySolverVisitor: LHS of differential equation is not a PrimeName");
        return;
    }

    auto locals = py::dict("equation_string"_a = nmodl::to_nmodl(node),
                           "t_var"_a = codegen::naming::NTHREAD_T_VARIABLE,
                           "dt_var"_a = codegen::naming::NTHREAD_DT_VARIABLE, "vars"_a = vars);
    py::exec(R"(
            from nmodl.ode import integrate2c
            exception_message = ""
            try:
                solution = integrate2c(equation_string, t_var, dt_var, vars)
            except Exception as e:
                # if we fail, fail silently and return empty string
                solution = ""
                exception_message = str(e)
        )",
             py::globals(), locals);

    auto solution = locals["solution"].cast<std::string>();
    auto exception_message = locals["exception_message"].cast<std::string>();
    if (!exception_message.empty()) {
        logger->warn("SympySolverVisitor: python exception: " + exception_message);
    }
    if (!solution.empty()) {
        auto statement = create_statement(solution);
        auto expr_statement = std::dynamic_pointer_cast<ExpressionStatement>(statement);
        auto bin_expr = std::dynamic_pointer_cast<BinaryExpression>(
            expr_statement->get_expression());
        lhs.reset(bin_expr->lhs->clone());
        rhs.reset(bin_expr->rhs->clone());
    } else {
        logger->warn("SympySolverVisitor: analytic solution to differential equation not possible");
    }
}

void SympySolverVisitor::visit_derivative_block(ast::DerivativeBlock* node) {
    // get any local vars
    auto symtab = node->get_statement_block()->get_symbol_table();
    if (symtab) {
        auto localvars = symtab->get_variables_with_properties(NmodlType::local_var);
        for (auto v: localvars) {
            vars.insert(v->get_name());
        }
    }
    node->visit_children(this);
}

void SympySolverVisitor::visit_program(ast::Program* node) {
    vars = get_global_vars(node);
    node->visit_children(this);
}
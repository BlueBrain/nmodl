/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::NeuronSolveVisitor
 */

#include <map>
#include <string>

#include "symtab/decl.hpp"
#include "visitors/ast_visitor.hpp"


namespace nmodl {
namespace visitor {

/**
 * \addtogroup solver
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class NeuronSolveVisitor
 * \brief %Visitor that solves ODEs using old solvers of NEURON
 *
 * This pass solves ODEs in derivative block using `cnexp`, `euler` and
 * `derivimplicit`method. This solved mimics original implementation in
 * nocmodl/mod2c. The original ODEs get replaced with the solution and
 * transformations are performed at AST level.
 *
 * \sa nmodl::visitor::SympySolverVisitor
 */
class NeuronSolveVisitor: public AstVisitor {
  private:
    /// true while visiting differential equation
    bool differential_equation = false;

    /// global symbol table
    symtab::SymbolTable* program_symtab = nullptr;

    /// a map holding solve block names and methods
    std::map<std::string, std::string> solve_blocks;

    /// method specified in solve block
    std::string solve_method;

    /// visiting derivative block
    bool derivative_block = false;

    /// the derivative name currently being visited
    std::string derivative_block_name;

    std::vector<std::shared_ptr<ast::Statement>> euler_solution_expressions;

  public:
    NeuronSolveVisitor() = default;

    void visit_solve_block(ast::SolveBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_derivative_block(ast::DerivativeBlock& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;
    void visit_program(ast::Program& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

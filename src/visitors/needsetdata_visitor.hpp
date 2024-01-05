/*
 * Copyright 2024 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::NeedSetDataVisitor
 */

#include <set>
#include <stack>

#include "ast/block.hpp"
#include "symtab/decl.hpp"
#include "visitors/ast_visitor.hpp"


namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class NeedSetDataVisitor
 * \brief %Visitor for figuring out if function and procedure need setdata call
 *
 */
class NeedSetDataVisitor: public ConstAstVisitor {
  private:
    /// program symbol table
    symtab::SymbolTable* psymtab = nullptr;

    std::stack<const ast::Block*> function_or_procedure_stack;

    /// function or procedure has RANGE or POINTER variable or calls
    /// function or procedure that has one of those and needs to throw
    /// if setdata() is not priorly called
    std::unordered_set<const ast::Block*> function_proc_need_setdata;

  public:

    void visit_var_name(const ast::VarName& node) override;

    void visit_function_call(const ast::FunctionCall& node) override;

    void visit_function_block(const ast::FunctionBlock& node) override;

    void visit_procedure_block(const ast::ProcedureBlock& node) override;

    void visit_program(const ast::Program& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

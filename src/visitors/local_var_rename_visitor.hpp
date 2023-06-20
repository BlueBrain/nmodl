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
 * \brief \copybrief nmodl::visitor::LocalVarRenameVisitor
 */

#include <map>
#include <stack>

#include "symtab/decl.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class LocalVarRenameVisitor
 * \brief %Visitor to rename local variables conflicting with global scope
 *
 * Motivation: During inlining we have to do data-flow-analysis. Consider
 * below example:
 *
 * \code{.mod}
 *      NEURON {
 *          RANGE tau, beta
 *      }
 *
 *      DERIVATIVE states() {
 *          LOCAL tau
 *          ...
 *          rates()
 *      }
 *
 *      PROCEDURE rates() {
 *          tau = beta * 0.12 * some_var
 *      }
 * \endcode
 *
 * When rates() will be inlined into states(), local definition of tau will
 * conflict with range variable tau. Hence we can't just copy the statements.
 * Dataflow analysis could be done at the time of inlining. Other way is to run
 * this pass before inlining and pre-rename any local-global variable conflicts.
 * As we are renaming local variables only, it's safe and there are no side effects.
 *
 * \todo
 *   - Currently we are renaming variables even if there is no inlining candidates.
 *     In this case ideally we should not rename.
 */
class LocalVarRenameVisitor: public AstVisitor {
  private:
    /// non-null symbol table in the scope hierarchy
    const symtab::SymbolTable* symtab = nullptr;

    /// symbol tables in case of nested blocks
    std::stack<const symtab::SymbolTable*> symtab_stack;

    /// variables currently being renamed and their count
    std::map<std::string, int> renamed_variables;

  public:
    LocalVarRenameVisitor() = default;
    void visit_statement_block(ast::StatementBlock& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

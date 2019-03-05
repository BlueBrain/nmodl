/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <vector>

#include "ast/ast.hpp"
#include "visitors/nmodl_visitor.hpp"


namespace nmodl {

/**
 * \class SympyNmodlPrintVisitor
 * \brief Customized Ast to NMODL printer for SymPy analysis
 *
 * Default NmodlPrintVisitor convert every node to corresponding
 * NMODL constructs. For SymPy we would like to ignore certain constructs
 * in the AST like units. This printer ignore such types and convert rest
 * to NMODL.
 */
class SympyNmodlPrintVisitor: public NmodlPrintVisitor {
  public:
    void visit_unit(ast::Unit* /* node */) override{};

    SympyNmodlPrintVisitor(std::ostream& stream)
        : NmodlPrintVisitor(stream) {}
};


static inline std::string to_sympy_nmodl(ast::AST* node) {
    std::stringstream stream;
    SympyNmodlPrintVisitor v(stream);
    node->accept(&v);
    return stream.str();
}

}  // namespace nmodl
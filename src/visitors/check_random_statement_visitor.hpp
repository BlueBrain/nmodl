/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <map>
#include <string>

#include "visitors/ast_visitor.hpp"

/**
 * \file
 * \brief \copybrief nmodl::visitor::CheckRandomStatementVisitor
 */

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

class CheckRandomStatementVisitor: protected ConstAstVisitor  {
  private:
    void visit_random(const ast::Random& node) override;

    const std::map<std::string, int> distributions = {
        {"UNIFORM", 2},
        {"NEGEXP", 1},
        {"NORMAL", 2},
    };

  public:
    CheckRandomStatementVisitor() = default;

    void visit_program(const ast::Program& node) override;
};

/**
 * \}
 */

} // namespace visitor
} // namespace nmodl

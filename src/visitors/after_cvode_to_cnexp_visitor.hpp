/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::AfterCVodeToCnexpVisitor
 */

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class AfterCVodeToCnexpVisitor
 * \brief Visitor to change usage of after_cvode solver to cnexp
 *

 */

class AfterCVodeToCnexpVisitor: public AstVisitor {
  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    AfterCVodeToCnexpVisitor() = default;

    /// \}
    void visit_solve_block(ast::SolveBlock& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

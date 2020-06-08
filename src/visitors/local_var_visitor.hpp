/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::LocalToAssignedVisitor
 */

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class LocalToAssignedVisitor
 * \brief Visitor to convert top level LOCAL variables to ASSIGNED variables
 *
 * Some of the existing mod file include declaration of LOCAL variables in
 * the top level of the mod file. Those variables are normally written in
 * the INITIAL block which is executed potentially by multiple threads.
 * This commands them to become ASSIGNED variables which by default are
 * handled as RANGE variables to be able to be written and read by multiple
 * threads without race conditions.
 * For example:
 *
 * \code{.mod}
 *      NEURON {
 *          SUFFIX test
 *  	    GLOBAL x
 *      }
 *      LOCAL x, y
 *      INITIAL {
 *          x = 1
 *      }
 * \endcode
 */

class LocalToAssignedVisitor: public AstVisitor {
  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    LocalToAssignedVisitor() = default;

    /// \}

    /// Visit ast::Program node to transform top level LOCAL variables
    /// to ASSIGNED if they are written in the mod file
    void visit_program(ast::Program& node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

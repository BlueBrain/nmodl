/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::LocalToRangeVisitor
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
 * \class LocalToRangeVisitor
 * \brief Visitor to convert top level LOCAL variables to RANGE variables
 */

class LocalToRangeVisitor: public AstVisitor {
public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    LocalToRangeVisitor() = default;

    /// \}

    /// Visit ast::Program node to transform top level LOCAL variables
    /// to RANGE if they are written in the mod file
    void visit_program(ast::Program& node) override;
};

/** @} */  // end of visitor_classes

    }  // namespace visitor
}  // namespace nmodl

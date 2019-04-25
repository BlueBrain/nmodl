/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once


namespace nmodl {
/// Implementation of different AST visitors
namespace visitor {

/**
 * @defgroup visitor Visitor Implementation
 * @brief All visitors related implementation details
 *
 * @defgroup visitor_classes Visitors
 * @ingroup visitor
 * @brief Different visitors implemented in NMODL
 * @{
 */

/**
 * \brief Abstract base class for all visitors implementation
 *
 * This class defines interface for all concrete visitors implementation.
 * Note that this class only provides interface that could be implemented
 * by oncrete visitors like ast::AstVisitor.
 *
 * \sa ast::AstVisitor
 */
class Visitor {

    public:
        virtual ~Visitor() = default;

        {% for node in nodes %}
        /// visit node of type ast::{{ node.class_name }}
        virtual void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}* node) = 0;
        {% endfor %}
};

}  // namespace visitor
}  // namespace nmodl

/** @} */  // end of visitor_classes
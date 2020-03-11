/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 *
 * \dir
 * \brief Auto generated visitors
 *
 * \file
 * \brief \copybrief nmodl::visitor::CkParentVisitor
 */

#include "visitors/ast_visitor.hpp"
#include <stack>


namespace nmodl {
namespace visitor {

/**
 * @ingroup visitor_classes
 * @{
 */

/**
 * \brief Visitor that checks the parents
 */
class CkParentVisitor : public AstVisitor {
    private:
        /**
        * \brief Keeps track of the parents while going down the tree
        */
        ast::Ast* parent = nullptr;
        bool ckRootParentNull = false;
    public:

        CkParentVisitor(const bool ckRootParentNull = false) {}

        int lookup(ast::Ast* node);

        {% for node in nodes %}
        /**
        * \brief Go through the tree while checking the parents
        */
        void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}* node) override;
        {% endfor %}
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
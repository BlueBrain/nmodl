/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/ckparent_visitor.hpp"


namespace nmodl {
namespace visitor {

using namespace ast;

{% for node in nodes %}
void CkParentVisitor::visit_{{ node.class_name|snake_case }}({{ node.class_name }}* node) {

    // check that the root node exists, probably superfluous
    if (!node) {
        return;
    }

    // throw an exception if parents do not match with the line of succession
    if (lineOfSuccession.empty() && node->get_parent() ||
        lineOfSuccession.top() != node->get_parent()) {
        //TODO throw exception
    }

    // add this node to the stack, it is the parent to check when visiting children
    lineOfSuccession.push(node);

    // visit its children
    node->visit_children(*this);

    // remove this node from the stack: it is not the parent of its siblings
    lineOfSuccession.pop();
}

{% endfor %}

}  // namespace visitor
}  // namespace nmodl
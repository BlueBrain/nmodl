/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/ckparent_visitor.hpp"
#include <sstream>
#include <string>


namespace nmodl {
namespace visitor {

using namespace ast;

int CkParentVisitor::lookup(Ast* node) {

    // There is no clear() in std::stack...
    while ( ! lineOfSuccession.empty() ){
        lineOfSuccession.pop();
    }

    node->accept(*this);

    return 0;
}

void modToken2error(std::stringstream& ss, Ast* node) {
    ss << "\n";
    if (!node) {
        ss << "nullptr\n";
        return;
    }
    if (!node->get_token()) {
        ss << "noToken,type: \n";
        ss << static_cast<int>(node->get_node_type()) << "\n";
        return;
    }
    ss << (*node->get_token()) << '\n';
}


{% for node in nodes %}
void CkParentVisitor::visit_{{ node.class_name|snake_case }}({{ node.class_name }}* node) {

    // check that the root node exists, probably superfluous
    if (!node) {
        return;
    }

    if (lineOfSuccession.empty()) {
        if (ckRootParentNull && node->get_parent()) {
            std::stringstream ss;

            ss << "root->parent ";
            modToken2error(ss, node->get_parent());
            ss << " is set when it should be nullptr";

            throw std::runtime_error(ss.str());
        }
    }
    else {
        if (lineOfSuccession.top() != node->get_parent()) {
            std::stringstream ss;

            ss << "parent: ";
            modToken2error(ss, lineOfSuccession.top());
            ss << "and child->parent: ";
            modToken2error(ss, node->get_parent());
            ss << "missmatch";

            throw std::runtime_error(ss.str());
        }
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
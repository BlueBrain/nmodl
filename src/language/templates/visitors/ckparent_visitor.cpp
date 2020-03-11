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
    parent = nullptr;

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

    if (!parent) {
        if (ckRootParentNull && node->get_parent()) {
            std::stringstream ss;

            ss << "visit_{{ node.class_name|snake_case }}\n";
            ss << "root->parent ";
            modToken2error(ss, node->get_parent());
            ss << " is set when it should be nullptr";

            throw std::runtime_error(ss.str());
        }
    }
    else {
        if (parent != node->get_parent()) {
            std::stringstream ss;

            ss << "visit_{{ node.class_name|snake_case }}\n";
            ss << "parent: ";
            modToken2error(ss, parent);
            ss << "and child->parent: ";
            modToken2error(ss, node->get_parent());
            ss << "missmatch";

            throw std::runtime_error(ss.str());
        }
    }

    // Now, this node is the parent. I go down the tree
    parent = node;

    // visit its children
    node->visit_children(*this);

    // I am done with these children, I go up the tree. The parent of this node is the parent
    parent = node->get_parent();
}

{% endfor %}

}  // namespace visitor
}  // namespace nmodl
/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

///
/// THIS FILE IS GENERATED AT BUILD TIME AND SHALL NOT BE EDITED.
///

#include "visitors/ast_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/json_visitor.hpp"


namespace nmodl {
namespace visitor {

using namespace ast;

{% for node in nodes %}
void AstVisitor::visit_{{ node.class_name|snake_case }}({{ node.class_name }}& node) {
    std::cerr << "visiting {{node.class_name}}"  << std::endl;
    std::stringstream ssb;
    nmodl::visitor::JSONVisitor vb(ssb);
    vb.compact_json(true);
    node.accept(vb);
    vb.flush();
    std::cerr << ssb.str() << std::endl;
    std::cerr << "node: " << *node << " node parent:" << node.get_parent() << std::endl;
    node.visit_children(*this);
    std::cerr << "back into {{node.class_name}}"  << std::endl;
    std::stringstream ssa;
    nmodl::visitor::JSONVisitor va(ssa);
    va.compact_json(true);
    node.accept(va);
    va.flush();
    std::cerr << ssa.str() << std::endl;
}
{% endfor %}

{% for node in nodes %}
void ConstAstVisitor::visit_{{ node.class_name|snake_case }}(const {{ node.class_name }}& node) {
    node.visit_children(*this);
}
{% endfor %}

}  // namespace visitor
}  // namespace nmodl


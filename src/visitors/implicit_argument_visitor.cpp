/*************************************************************************
 * Copyright (C) 2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/
#include "visitors/implicit_argument_visitor.hpp"
#include "ast/function_call.hpp"
#include "ast/string.hpp"

#include <cassert>
#include <iostream>

namespace nmodl {
namespace visitor {

// TODO should this also handle the case covered by
// needs_neuron_thread_first_arg?
void ImplicitArgumentVisitor::visit_function_call(ast::FunctionCall& node) {
    auto function_name = node.get_node_name();
    auto const& arguments = node.get_arguments();
    if (function_name == "nrn_ghk") {
        // This function is traditionally used in MOD files with four arguments, but
        // its value also depends on the global celsius variable so the real
        // function in CoreNEURON has a 5th argument for that.
        if (arguments.size() == 4) {
            auto new_arguments = arguments;
            new_arguments.insert(new_arguments.end(), std::make_shared<ast::String>("celsius"));
            node.set_arguments(std::move(new_arguments));
        }
    }
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

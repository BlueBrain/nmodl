/*************************************************************************
 * Copyright (C) 2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/
#include "visitors/implicit_argument_visitor.hpp"
#include "ast/function_call.hpp"
#include "ast/string.hpp"
#include "lexer/token_mapping.hpp"

#include <cassert>
#include <iostream>

namespace nmodl {
namespace visitor {

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
    } else if (nmodl::details::needs_neuron_thread_first_arg(function_name)) {
        // We need to insert `nt` as the first argument if it's not already
        // there
        // TODO: add a test where the first argument to at_time as some `a+b`
        // expression and re-add extra logic based on get_node_type() and
        // AstNodeType::NAME, AstNodeType::STRING, AstNodeType::CONSTANT_VAR,
        // AstNodeType::VAR_NAME, AstNodeType::LOCAL_VAR.
        if (arguments.empty() || arguments.front()->get_node_name() != "nt") {
            auto new_arguments = arguments;
            new_arguments.insert(new_arguments.begin(), std::make_shared<ast::String>("nt"));
            node.set_arguments(std::move(new_arguments));
        }
    }
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

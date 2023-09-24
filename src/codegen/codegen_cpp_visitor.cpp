/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "codegen/codegen_cpp_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/rename_visitor.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

using visitor::RenameVisitor;

template <typename T>
bool CodegenCppVisitor::has_parameter_of_name(const T& node, const std::string& name) {
    auto parameters = node->get_parameters();
    return std::any_of(parameters.begin(),
                       parameters.end(),
                       [&name](const decltype(*parameters.begin()) arg) {
                           return arg->get_node_name() == name;
                       });
}


/**
 * \todo Issue with verbatim renaming. e.g. pattern.mod has info struct with
 * index variable. If we use "index" instead of "indexes" as default argument
 * then during verbatim replacement we don't know the index is which one. This
 * is because verbatim renaming pass has already stripped out prefixes from
 * the text.
 */
void CodegenCppVisitor::rename_function_arguments() {
    const auto& default_arguments = stringutils::split_string(nrn_thread_arguments(), ',');
    for (const auto& dirty_arg: default_arguments) {
        const auto& arg = stringutils::trim(dirty_arg);
        RenameVisitor v(arg, "arg_" + arg);
        for (const auto& function: info.functions) {
            if (has_parameter_of_name(function, arg)) {
                function->accept(v);
            }
        }
        for (const auto& function: info.procedures) {
            if (has_parameter_of_name(function, arg)) {
                function->accept(v);
            }
        }
    }
}

}  // namespace codegen
}  // namespace nmodl
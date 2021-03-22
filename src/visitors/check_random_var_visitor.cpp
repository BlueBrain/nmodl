/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "check_random_var_visitor.hpp"

#include <utility>

#include "ast/function_call.hpp"
#include "ast/random_var.hpp"
#include "ast/wrapped_expression.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

using namespace fmt::literals;

void CheckRandomVarVisitor::visit_program(const ast::Program& node) {
    ConstAstVisitor::visit_program(node);
}


void CheckRandomVarVisitor::visit_random_var(const ast::RandomVar& node) {
    logger->info(to_nmodl(node));
    auto func_name = static_cast<ast::FunctionCall*>(node.get_distribution()->get_expression().get())->get_name();
    int nargs = static_cast<ast::FunctionCall*>(node.get_distribution()->get_expression().get())->get_arguments().size();
    if (distributions.find(func_name) == distributions.end()) {
        logger->error("{} is not a valid pdf"_format(func_name));
    }
}

}  // namespace visitor
}  // namespace nmodl

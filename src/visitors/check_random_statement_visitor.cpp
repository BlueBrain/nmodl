/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "check_random_statement_visitor.hpp"

#include <utility>

#include "ast/function_call.hpp"
#include "ast/name.hpp"
#include "ast/random.hpp"
#include "ast/wrapped_expression.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

using namespace fmt::literals;

void CheckRandomStatementVisitor::visit_program(const ast::Program& node) {
    ConstAstVisitor::visit_program(node);
}


void CheckRandomStatementVisitor::visit_random(const ast::Random& node) {
    logger->info(to_nmodl(node));
    auto& distribution = node.get_distribution();
    auto distribution_name = distribution->get_node_name();
    auto& params = node.get_distribution_params();
    if (distributions.find(distribution_name) != distributions.end()) {
        if (distributions.at(distribution_name) != params.size()) {
            throw std::logic_error("Validation Error: {} declared with {} instead of {} parameters"_format(distribution_name, params.size()
                                                                                                               , distributions.at(distribution_name)));
        }
    } else {
        throw std::logic_error("Validation Error: distribution {} unknown"_format(distribution_name));
    }
}

}  // namespace visitor
}  // namespace nmodl

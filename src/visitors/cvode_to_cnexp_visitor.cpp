/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/cvode_to_cnexp_visitor.hpp"

#include "ast/name.hpp"
#include "ast/solve_block.hpp"
#include "ast/string.hpp"
#include "codegen/codegen_naming.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

void CVodeToCnexpVisitor::visit_solve_block(ast::SolveBlock& node) {
    auto method = node.get_method();
    if (method->get_node_name() == codegen::naming::CVODE_METHOD) {
        logger->warn("CVode solver of {} in {} replaced with cnexp solver",
                     node.get_block_name()->get_node_name(),
                     method->get_token()->position());
        node.set_method(std::make_shared<ast::Name>(
            std::make_shared<ast::String>(codegen::naming::CNEXP_METHOD)));
    }
}

}  // namespace visitor
}  // namespace nmodl

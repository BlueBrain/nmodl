/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "visitors/after_cvode_to_cnexp_visitor.hpp"

#include "ast/name.hpp"
#include "ast/solve_block.hpp"
#include "ast/string.hpp"
#include "codegen/codegen_naming.hpp"
#include "utils/logger.hpp"

namespace nmodl {
namespace visitor {

void AfterCVodeToCnexpVisitor::visit_solve_block(ast::SolveBlock& node) {
    const auto& method = node.get_method();
    if (method != nullptr && method->get_node_name() == codegen::naming::AFTER_CVODE_METHOD) {
        logger->warn("CVode solver of {} in {} replaced with cnexp solver",
                     node.get_block_name()->get_node_name(),
                     method->get_token()->position());
        node.set_method(std::make_shared<ast::Name>(
            std::make_shared<ast::String>(codegen::naming::CNEXP_METHOD)));
    }
}

}  // namespace visitor
}  // namespace nmodl

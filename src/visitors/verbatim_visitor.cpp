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

#include "visitors/verbatim_visitor.hpp"

#include <iostream>

#include "ast/string.hpp"
#include "ast/verbatim.hpp"

namespace nmodl {
namespace visitor {

void VerbatimVisitor::visit_verbatim(const ast::Verbatim& node) {
    std::string block;
    const auto& statement = node.get_statement();
    if (statement) {
        block = statement->eval();
    }
    if (!block.empty() && verbose) {
        std::cout << "BLOCK START" << block << "\nBLOCK END \n\n";
    }

    blocks.push_back(block);
}

}  // namespace visitor
}  // namespace nmodl

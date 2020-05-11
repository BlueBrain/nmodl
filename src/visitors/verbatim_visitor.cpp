/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/verbatim_visitor.hpp"
#include <iostream>


namespace nmodl {
namespace visitor {

void VerbatimVisitor::visit_verbatim(ast::Verbatim& node) {
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

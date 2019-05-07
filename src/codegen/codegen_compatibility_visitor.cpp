/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <algorithm>
#include <string>

#include "codegen/codegen_compatibility_visitor.hpp"
#include "utils/logger.hpp"
#include "visitors/lookup_visitor.hpp"

namespace nmodl {
namespace codegen {

using visitor::AstLookupVisitor;

/**
 * \details Checks all the ast::AstNodeTypes that are not compatible with NMODL code
 * generation and prints related messages. If there is some kind of incompatibility
 * stop NMODL code generation.
 */
bool CodegenCompatibilityVisitor::find_incompatible_ast_nodes(Ast* node) {
    std::vector<AstNodeType> incompatible_ast_types = {AstNodeType::SOLVE_BLOCK,
                                                       AstNodeType::TERMINAL_BLOCK,
                                                       AstNodeType::PARTIAL_BLOCK,
                                                       AstNodeType::MATCH_BLOCK,
                                                       AstNodeType::BA_BLOCK,
                                                       AstNodeType::CONSTANT_BLOCK,
                                                       AstNodeType::CONSTRUCTOR_BLOCK,
                                                       AstNodeType::DESTRUCTOR_BLOCK,
                                                       AstNodeType::DISCRETE_BLOCK,
                                                       AstNodeType::FUNCTION_TABLE_BLOCK,
                                                       AstNodeType::INDEPENDENT_BLOCK,
                                                       AstNodeType::GLOBAL_VAR,
                                                       AstNodeType::POINTER_VAR,
                                                       AstNodeType::BBCORE_POINTER_VAR};
    incompatible_ast_nodes = AstLookupVisitor().lookup(node, incompatible_ast_types);

    std::stringstream ss;
    for (auto it: incompatible_ast_nodes) {
        if (it->is_solve_block()) {
            auto solve_block = dynamic_cast<ast::SolveBlock*>(it.get());
            auto method = solve_block->get_method();
            if (method != nullptr && method->get_node_name() != "cnexp" &&
                method->get_node_name() != "euler" && method->get_node_name() != "derivimplicit" &&
                method->get_node_name() != "sparse") {
                ss << method->get_node_name() << " solving method not supported\n";
                ss << "Supported methods are cnexp, euler, derivimplicit and sparse\n";
            }
        } else if (it->is_terminal_block() || it->is_match_block() || it->is_discrete_block()) {
            std::string node_type_name = it->get_node_type_name();
            // remove "Block" substring
            node_type_name.erase(node_type_name.end() - 5, node_type_name.end());
            // turn name of node type to capital, as it is in mod files
            std::transform(node_type_name.begin(),
                           node_type_name.end(),
                           node_type_name.begin(),
                           ::toupper);

            ss << it->get_node_name() << " " << node_type_name << " construct is not supported\n";
        } else if (it->is_ba_block()) {
            ss << "BEFORE AFTER construct is not supported\n";
        } else if (it->is_constant_block() || it->is_constructor_block() ||
                   it->is_destructor_block()) {
            std::string node_type_name = it->get_node_type_name();
            // remove "Block" substring
            node_type_name.erase(node_type_name.end() - 5, node_type_name.end());
            // turn name of node type to capital, as it is in mod files
            std::transform(node_type_name.begin(),
                           node_type_name.end(),
                           node_type_name.begin(),
                           ::toupper);

            ss << node_type_name << " construct is not supported\n";
        } else if (it->is_independent_block()) {
            ss << "INDEPENDENT construct is not supported\n";
        } else if (it->is_partial_block()) {
            ss << it->get_node_name() << " PARTIAL construct found at ";
            ss << it->get_token()->position() << " is not supported\n";
        } else if (it->is_function_table_block()) {
            ss << it->get_node_name() << " FUNCTION_TABLE construct found at ";
            ss << it->get_token()->position() << " is not supported\n";
        } else if (it->is_global_var()) {
            auto global_var = dynamic_cast<ast::GlobalVar*>(it.get());
            if (node->get_symbol_table()->lookup(global_var->get_node_name())->get_write_count() >
                0) {
                ss << global_var->get_node_name();
                ss << " should be defined as a RANGE variable instead of GLOBAL to enable backend "
                      "transformations\n";
            }
        }
    }
    if (!ss.str().empty()) {
        logger->error("Code incompatibility detected");
        logger->error("Cannot translate mod file to .cpp file");
        logger->error("Fix the following errors and try again");
        std::cerr << ss.str();
        return 1;
    }
    return 0;
}

}  // namespace codegen
}  // namespace nmodl

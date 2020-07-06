/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/rename_visitor.hpp"

#include "ast/all.hpp"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {
namespace visitor {

/// rename matching variable
void RenameVisitor::visit_name(ast::Name& node) {
    const auto& name = node.get_node_name();
    std::string new_name;
    if (std::regex_match(name, regex)) {
        auto& value = node.get_value();
        /// Check if variable is already renamed and use the same naming otherwise add the new_name
        /// to the renamed_variables map
        if (add_random_suffix) {
            if (renamed_variables.find(name) != renamed_variables.end()) {
                new_name = renamed_variables[name];
            } else {
                const auto& vars = get_global_vars(*ast);
                if (add_prefix) {
                    new_name = suffix_random_string(vars, new_var_name_prefix + name);
                } else {
                    new_name = suffix_random_string(vars, new_var_name);
                }
                renamed_variables[name] = new_name;
            }
        } else {
            if (add_prefix) {
                new_name = new_var_name_prefix + name;
            } else {
                new_name = new_var_name;
            }
        }
        value->set(new_name);
        logger->warn("RenameVisitor :: Renaming variable {} in {} to {}",
                     name,
                     node.get_token()->position(),
                     new_name);
    }
}

/** Prime name has member order which is an integer. In theory
 * integer could be "macro name" and hence could end-up renaming
 * macro. In practice this won't be an issue as we order is set
 * by parser. To be safe we are only renaming prime variable.
 */
void RenameVisitor::visit_prime_name(ast::PrimeName& node) {
    node.visit_children(*this);
}

/**
 * Parse verbatim blocks and rename variable if it is used.
 */
void RenameVisitor::visit_verbatim(ast::Verbatim& node) {
    if (!rename_verbatim) {
        return;
    }

    const auto& statement = node.get_statement();
    auto text = statement->eval();
    parser::CDriver driver;

    driver.scan_string(text);
    auto tokens = driver.all_tokens();

    std::string result;
    for (auto& token: tokens) {
        if (std::regex_match(token, regex)) {
            /// Check if variable is already renamed and use the same naming otherwise add the
            /// new_name to the renamed_variables map
            std::string new_name;
            if (add_random_suffix) {
                if (renamed_variables.find(token) != renamed_variables.end()) {
                    new_name = renamed_variables[token];
                } else {
                    const auto& vars = get_global_vars(*ast);
                    if (add_prefix) {
                        new_name = suffix_random_string(vars, new_var_name_prefix + token);
                    } else {
                        new_name = suffix_random_string(vars, new_var_name);
                    }
                    renamed_variables[token] = new_name;
                }
            } else {
                if (add_prefix) {
                    new_name = new_var_name_prefix + token;
                } else {
                    new_name = new_var_name;
                }
            }
            result += new_name;
            logger->warn("RenameVisitor :: Renaming variable {} in VERBATIM block to {}",
                         token,
                         new_name);
        } else {
            result += token;
        }
    }
    statement->set(result);
}

}  // namespace visitor
}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/ispc_rename_visitor.hpp"

#include "ast/all.hpp"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {
namespace visitor {

/// rename matching variable
void IspcRenameVisitor::visit_name(ast::Name& node) {
    const auto& name = node.get_node_name();
    if (std::regex_match(name, double_regex)) {
        auto& value = node.get_value();
        const auto& vars = get_global_vars(*ast);
        const auto& new_name = suffix_random_string(vars, new_var_name_prefix + name);
        value->set(new_name);
        logger->warn("IspcRenameVisitor :: Renaming variable {} in {} to {}",
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
void IspcRenameVisitor::visit_prime_name(ast::PrimeName& node) {
    node.visit_children(*this);
}

/**
 * Parse verbatim blocks and rename variable if it is used.
 */
void IspcRenameVisitor::visit_verbatim(ast::Verbatim& node) {
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
        if (std::regex_match(token, double_regex)) {
            const auto& new_name = suffix_random_string(get_global_vars(*ast),
                                                        new_var_name_prefix + token);
            result += new_name;
            logger->warn("IspcRenameVisitor :: Renaming variable {} in VERBATIM block {} to {}",
                         token,
                         node.get_token()->position(),
                         new_name);
        } else {
            result += token;
        }
    }
    statement->set(result);
}

}  // namespace visitor
}  // namespace nmodl

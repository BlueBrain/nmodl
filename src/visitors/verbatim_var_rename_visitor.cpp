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

#include "visitors/verbatim_var_rename_visitor.hpp"

#include "ast/statement_block.hpp"
#include "ast/string.hpp"
#include "ast/verbatim.hpp"
#include "parser/c11_driver.hpp"
#include "src/utils/logger.hpp"


namespace nmodl {
namespace visitor {

void VerbatimVarRenameVisitor::visit_statement_block(ast::StatementBlock& node) {
    if (node.get_statements().empty()) {
        return;
    }

    auto current_symtab = node.get_symbol_table();
    if (current_symtab != nullptr) {
        symtab = current_symtab;
    }

    // some statements like from, while are of type expression statement type.
    // These statements contain statement block but do not have symbol table. And hence
    // we push last non-null symbol table on the stack.
    symtab_stack.push(symtab);

    // first need to process all children : perform recursively from innermost block
    for (const auto& item: node.get_statements()) {
        item->accept(*this);
    }

    /// go back to previous block in hierarchy
    symtab = symtab_stack.top();
    symtab_stack.pop();
}

/**
 * Rename variable used in verbatim block if defined in NMODL scope
 *
 * Check if variable is candidate for renaming and check if it is
 * defined in the nmodl blocks. If so, return "original" name of the
 * variable.
 */
std::string VerbatimVarRenameVisitor::rename_variable(const std::string& name) {
    bool rename_plausible = false;
    auto new_name = name;
    if (name.find(LOCAL_PREFIX) == 0) {
        new_name.erase(0, 2);
        rename_plausible = true;
    }
    if (name.find(RANGE_PREFIX) == 0) {
        new_name.erase(0, 3);
        rename_plausible = true;
    }
    if (name.find(ION_PREFIX) == 0) {
        new_name.erase(0, 1);
        rename_plausible = true;
    }
    if (rename_plausible) {
        auto symbol = symtab->lookup_in_scope(new_name);
        if (symbol != nullptr) {
            return new_name;
        }
        logger->warn("could not find {} definition in nmodl", name);
    }
    return name;
}


/**
 * Parse verbatim blocks and rename variables used
 */
void VerbatimVarRenameVisitor::visit_verbatim(ast::Verbatim& node) {
    const auto& statement = node.get_statement();
    const auto& text = statement->eval();
    parser::CDriver driver;

    driver.scan_string(text);
    const auto& tokens = driver.all_tokens();

    std::ostringstream oss;
    for (const auto& token: tokens) {
        oss << rename_variable(token);
    }
    statement->set(oss.str());
}

}  // namespace visitor
}  // namespace nmodl

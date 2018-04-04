#include "visitors/verbatim_var_rename_visitor.hpp"
#include "parser/c11_driver.hpp"

using namespace ast;
using namespace symtab;

void VerbatimVarRenameVisitor::visit_statement_block(StatementBlock* node) {
    if (node->statements.empty()) {
        return;
    }

    auto current_symtab = node->get_symbol_table();
    if (current_symtab != nullptr) {
        symtab = current_symtab;
    }

    // some statements like forall, from, while are of type expression statement type.
    // These statements contain statement block but do not have symbol table. And hence
    // we push last non-null symbol table on the stack.
    symtab_stack.push(symtab);

    // first need to process all children : perform recursively from innermost block
    for (auto& item : node->statements) {
        item->accept(this);
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
std::string VerbatimVarRenameVisitor::rename_variable(std::string name) {
    bool rename_plausible = false;
    auto new_name = name;
    if (name.find(local_prefix) == 0) {
        new_name.erase(0,2);
        rename_plausible = true;
    }
    if (name.find(range_prefix) == 0) {
        new_name.erase(0,3);
        rename_plausible = true;
    }
    if (rename_plausible) {
        auto symbol = symtab->lookup_in_scope(new_name);
        if (symbol != nullptr) {
            return new_name;
        }
        std::cerr << "Warning : could not find " << name << " definition in nmodl" << std::endl;
    }
    return name;
}


/**
 * Parse verbatim blocks and rename variables used
 */
void VerbatimVarRenameVisitor::visit_verbatim(Verbatim* node) {
    auto statement = node->get_statement();
    auto text = statement->eval();
    c11::Driver driver;

    driver.scan_string(text);
    auto tokens = driver.all_tokens();

    std::string result;
    for(auto& token: tokens) {
        result += rename_variable(token);
    }
    statement->set(result);
}

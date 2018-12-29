#include "visitors/rename_visitor.hpp"
#include "parser/c11_driver.hpp"

using namespace ast;

/// rename matching variable
void RenameVisitor::visit_name(Name* node) {
    std::string name = node->get_name();
    if (name == var_name) {
        node->value->set(new_var_name);
    }
}

/** Prime name has member order which is an integer. In theory
 * integer could be "macro name" and hence could end-up renaming
 * macro. In practice this won't be an issue as we order is set
 * by parser. To be safe we are only renaming prime variable.
 */
void RenameVisitor::visit_prime_name(ast::PrimeName* node) {
    node->visit_children(this);
}

/**
 * Parse verbatim blocks and rename variable if it is used.
 */
void RenameVisitor::visit_verbatim(Verbatim* node) {
    if (!rename_verbatim) {
        return;
    }

    auto statement = node->get_statement();
    auto text = statement->eval();
    c11::Driver driver;

    driver.scan_string(text);
    auto tokens = driver.all_tokens();

    std::string result;
    for (auto& token : tokens) {
        if (token == var_name) {
            result += new_var_name;
        } else {
            result += token;
        }
    }
    statement->set(result);
}

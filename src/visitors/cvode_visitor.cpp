/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/cvode_visitor.hpp"

#include "ast/all.hpp"
#include "lexer/token_mapping.hpp"
#include "pybind/pyembed.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"
#include <optional>
#include <utility>

namespace pywrap = nmodl::pybind_wrappers;

namespace nmodl {
namespace visitor {

static int get_index(const ast::IndexedName& node) {
    return std::stoi(to_nmodl(node.get_length()));
}

static void remove_conserve_statements(ast::StatementBlock& node) {
    auto conserve_equations = collect_nodes(node, {ast::AstNodeType::CONSERVE});
    if (!conserve_equations.empty()) {
        std::unordered_set<ast::Statement*> eqs;
        for (const auto& item: conserve_equations) {
            eqs.insert(std::dynamic_pointer_cast<ast::Statement>(item).get());
        }
        node.erase_statement(eqs);
    }
}

static std::pair<std::string, std::optional<int>> parse_independent_var(
    std::shared_ptr<ast::Identifier> node) {
    auto variable = std::make_pair(node->get_node_name(), std::optional<int>());
    if (node->is_indexed_name()) {
        variable.second = std::optional<int>(
            get_index(*std::dynamic_pointer_cast<const ast::IndexedName>(node)));
    }
    return variable;
}

static std::unordered_map<std::string, int> get_name_map(const ast::Expression& node,
                                                         const std::string& name) {
    std::unordered_map<std::string, int> name_map;
    // all of the "reserved" symbols
    auto reserved_symbols = get_external_functions();
    // all indexed vars
    auto indexed_vars = collect_nodes(node, {ast::AstNodeType::INDEXED_NAME});
    for (const auto& var: indexed_vars) {
        if (!name_map.count(var->get_node_name()) && var->get_node_name() != name &&
            std::none_of(reserved_symbols.begin(), reserved_symbols.end(), [&var](const auto item) {
                return var->get_node_name() == item;
            })) {
            logger->debug(
                "CvodeVisitor :: adding INDEXED_VARIABLE {} to "
                "node_map",
                var->get_node_name());
            name_map[var->get_node_name()] = get_index(
                *std::dynamic_pointer_cast<const ast::IndexedName>(var));
        }
    }
    return name_map;
}

static std::string cvode_set_lhs(ast::BinaryExpression& node) {
    const auto& lhs = node.get_lhs();

    auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();

    std::string varname;
    if (name->is_prime_name()) {
        varname = "D" + name->get_node_name();
        node.set_lhs(std::make_shared<ast::Name>(new ast::String(varname)));
    } else if (name->is_indexed_name()) {
        auto nodes = collect_nodes(*name, {ast::AstNodeType::PRIME_NAME});
        // make sure the LHS isn't just a plain indexed var
        if (!nodes.empty()) {
            varname = "D" + stringutils::remove_character(to_nmodl(node.get_lhs()), '\'');
            auto statement = fmt::format("{} = {}", varname, varname);
            auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
                create_statement(statement));
            const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
                expr_statement->get_expression());
            node.set_lhs(std::shared_ptr<ast::Expression>(bin_expr->get_lhs()->clone()));
        }
    }
    return varname;
}


class CvodeHelperVisitor: public AstVisitor {
  protected:
    symtab::SymbolTable* program_symtab = nullptr;
    bool in_differential_equation = false;
  public:
    inline void visit_diff_eq_expression(ast::DiffEqExpression& node) {
        in_differential_equation = true;
        node.visit_children(*this);
        in_differential_equation = false;
    }
};

class NonStiffVisitor: public CvodeHelperVisitor {
  public:
    explicit NonStiffVisitor(symtab::SymbolTable* symtab) {
        program_symtab = symtab;
    }

    inline void visit_binary_expression(ast::BinaryExpression& node) {
        const auto& lhs = node.get_lhs();

        if (!in_differential_equation || !lhs->is_var_name()) {
            return;
        }

        auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();
        auto varname = cvode_set_lhs(node);

        if (program_symtab->lookup(varname) == nullptr) {
            auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
            symbol->set_original_name(name->get_node_name());
            program_symtab->insert(symbol);
        }
    }
};

class StiffVisitor: public CvodeHelperVisitor {
  public:
    explicit StiffVisitor(symtab::SymbolTable* symtab) {
        program_symtab = symtab;
    }

    inline void visit_binary_expression(ast::BinaryExpression& node) {
        const auto& lhs = node.get_lhs();

        if (!in_differential_equation || !lhs->is_var_name()) {
            return;
        }

        auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();
        auto varname = cvode_set_lhs(node);

        if (program_symtab->lookup(varname) == nullptr) {
            auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
            symbol->set_original_name(name->get_node_name());
            program_symtab->insert(symbol);
        }

        auto rhs = node.get_rhs();
        // map of all indexed symbols (need special treatment in SymPy)
        auto name_map = get_name_map(*rhs, name->get_node_name());
        auto diff2c = pywrap::EmbeddedPythonLoader::get_instance().api().diff2c;
        auto [jacobian,
              exception_message] = diff2c(to_nmodl(*rhs), parse_independent_var(name), name_map);
        if (!exception_message.empty()) {
            logger->warn("CvodeVisitor :: python exception: {}", exception_message);
        }
        // NOTE: LHS can be anything here, the equality is to keep `create_statement` from
        // complaining, we discard the LHS later
        auto statement = fmt::format("{} = {} / (1 - dt * ({}))", varname, varname, jacobian);
        logger->debug("CvodeVisitor :: replacing statement {} with {}", to_nmodl(node), statement);
        auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
            create_statement(statement));
        const auto bin_expr = std::dynamic_pointer_cast<const ast::BinaryExpression>(
            expr_statement->get_expression());
        node.set_rhs(std::shared_ptr<ast::Expression>(bin_expr->get_rhs()->clone()));
    }
};


void CvodeVisitor::visit_program(ast::Program& node) {
    auto der_blocks = collect_nodes(node, {ast::AstNodeType::DERIVATIVE_BLOCK});
    if (!der_blocks.empty()) {
        auto der_block = std::dynamic_pointer_cast<ast::DerivativeBlock>(der_blocks[0]);

        auto non_stiff_block = der_block->get_statement_block()->clone();
        remove_conserve_statements(*non_stiff_block);

        auto stiff_block = der_block->get_statement_block()->clone();
        remove_conserve_statements(*stiff_block);

        NonStiffVisitor(node.get_symbol_table()).visit_statement_block(*non_stiff_block);
        StiffVisitor(node.get_symbol_table()).visit_statement_block(*stiff_block);
        auto prime_vars = collect_nodes(*der_block, {ast::AstNodeType::PRIME_NAME});
        node.emplace_back_node(new ast::CvodeBlock(
            der_block->get_name(),
            std::shared_ptr<ast::Integer>(new ast::Integer(prime_vars.size(), nullptr)),
            std::shared_ptr<ast::StatementBlock>(non_stiff_block),
            std::shared_ptr<ast::StatementBlock>(stiff_block)));
    }
}

}  // namespace visitor
}  // namespace nmodl

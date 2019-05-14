/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include "codegen/codegen_compatibility_visitor.hpp"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/lookup_visitor.hpp"

using namespace fmt::literals;

namespace nmodl {
namespace codegen {

using visitor::AstLookupVisitor;

std::string CodegenCompatibilityVisitor::return_error_if_solve_method_is_unhandled(
    std::shared_ptr<ast::SolveBlock>& solve_block_ast_node) {
    std::stringstream unhandled_method_error_message;
    auto method = solve_block_ast_node->get_method();
    if (method == nullptr)
        return "";
    auto unhandled_solver_method = handled_solvers.find(method->get_node_name()) ==
                                   handled_solvers.end();
    if (unhandled_solver_method) {
        unhandled_method_error_message
            << "\"{}\" solving method used at [{}] not handled. Supported methods are cnexp, euler, derivimplicit and sparse\n"_format(
                   method->get_node_name(), method->get_token()->position());
    }
    return unhandled_method_error_message.str();
}

template <typename T>
std::string CodegenCompatibilityVisitor::return_error_with_name(
    const std::shared_ptr<ast::Ast>& ast_node) {
    auto real_type_block = std::dynamic_pointer_cast<T>(ast_node);
    return "\"{}\" {}construct found at [{}] is not handled\n"_format(
        ast_node->get_node_name(),
        real_type_block->get_nmodl_name(),
        real_type_block->get_token()->position());
}

template <typename T>
std::string CodegenCompatibilityVisitor::return_error_without_name(
    const std::shared_ptr<ast::Ast>& ast_node) {
    auto real_type_block = std::dynamic_pointer_cast<T>(ast_node);
    return "{}construct found at [{}] is not handled\n"_format(
        real_type_block->get_nmodl_name(), real_type_block->get_token()->position());
}

std::string CodegenCompatibilityVisitor::return_error_global_var(
    Ast* node,
    std::shared_ptr<ast::GlobalVar>& global_var) {
    std::stringstream error_message_global_var;
    if (node->get_symbol_table()->lookup(global_var->get_node_name())->get_write_count() > 0) {
        error_message_global_var
            << "\"{}\" variable found at [{}] should be defined as a RANGE variable instead of GLOBAL to enable backend transformations\n"_format(
                   global_var->get_node_name(), global_var->get_token()->position());
    }
    return error_message_global_var.str();
}

std::string CodegenCompatibilityVisitor::return_error_pointer(
    std::shared_ptr<ast::PointerVar>& pointer_var) {
    return "\"{}\" POINTER found at [{}] should be defined as BBCOREPOINTER to use it in CoreNeuron\n"_format(
        pointer_var->get_node_name(), pointer_var->get_token()->position());
}

std::string CodegenCompatibilityVisitor::return_error_if_no_bbcore_read_write(Ast* node) {
    std::stringstream error_message_no_bbcore_read_write;
    auto verbatim_nodes = AstLookupVisitor().lookup(node, AstNodeType::VERBATIM);
    auto found_bbcore_read = false;
    auto found_bbcore_write = false;
    for (const auto& it: verbatim_nodes) {
        auto verbatim = std::dynamic_pointer_cast<ast::Verbatim>(it);

        auto verbatim_statement = verbatim->get_statement();
        auto verbatim_statement_string = verbatim_statement->get_value();

        // Parse verbatim block to find out if there is a token "bbcore_read" or
        // "bbcore_write". If there is not, then the function is not defined or
        // is commented.
        parser::CDriver driver;

        driver.scan_string(verbatim_statement_string);
        auto tokens = driver.all_tokens();

        for (const auto& token: tokens) {
            if (token == "bbcore_read") {
                found_bbcore_read = true;
            }
            if (token == "bbcore_write") {
                found_bbcore_write = true;
            }
        }
    }
    if (!found_bbcore_read) {
        error_message_no_bbcore_read_write
            << "\"bbcore_read\" function not defined in any VERBATIM block\n";
    }
    if (!found_bbcore_write) {
        error_message_no_bbcore_read_write
            << "\"bbcore_write\" function not defined in any VERBATIM block\n";
    }
    return error_message_no_bbcore_read_write.str();
}

/**
 * \details Checks all the ast::AstNodeType that are not handled in NMODL code
 * generation backend for CoreNEURON and prints related messages. If there is
 * some kind of incompatibility return false.
 */
bool CodegenCompatibilityVisitor::find_unhandled_ast_nodes(Ast* node) {
    unhandled_ast_nodes = AstLookupVisitor().lookup(node, unhandled_ast_types);

    std::stringstream ss;
    for (auto it: unhandled_ast_nodes) {
        auto node_type = it->get_node_type();
        switch (node_type) {
        case AstNodeType::SOLVE_BLOCK: {
            auto it_solve_block = std::dynamic_pointer_cast<ast::SolveBlock>(it);
            ss << return_error_if_solve_method_is_unhandled(it_solve_block);
        } break;
        case AstNodeType::DISCRETE_BLOCK:
            ss << return_error_with_name<DiscreteBlock>(it);
            break;
        case AstNodeType::PARTIAL_BLOCK:
            ss << return_error_with_name<PartialBlock>(it);
            break;
        case AstNodeType::BEFORE_BLOCK:
            ss << return_error_without_name<BeforeBlock>(it);
            break;
        case AstNodeType::AFTER_BLOCK:
            ss << return_error_without_name<AfterBlock>(it);
            break;
        case AstNodeType::MATCH_BLOCK:
            ss << return_error_without_name<MatchBlock>(it);
            break;
        case AstNodeType::CONSTANT_BLOCK:
            ss << return_error_without_name<ConstantBlock>(it);
            break;
        case AstNodeType::CONSTRUCTOR_BLOCK:
            ss << return_error_without_name<ConstructorBlock>(it);
            break;
        case AstNodeType::DESTRUCTOR_BLOCK:
            ss << return_error_without_name<DestructorBlock>(it);
            break;
        case AstNodeType::INDEPENDENT_BLOCK:
            ss << return_error_without_name<IndependentBlock>(it);
            break;
        case AstNodeType::FUNCTION_TABLE_BLOCK:
            ss << return_error_without_name<FunctionTableBlock>(it);
            break;
        case AstNodeType::GLOBAL_VAR: {
            auto it_global_var = std::dynamic_pointer_cast<ast::GlobalVar>(it);
            ss << return_error_global_var(node, it_global_var);
        } break;
        case AstNodeType::POINTER_VAR: {
            auto it_pointer_var = std::dynamic_pointer_cast<ast::PointerVar>(it);
            ss << return_error_pointer(it_pointer_var);
        } break;
        case AstNodeType::BBCORE_POINTER_VAR:
            ss << return_error_if_no_bbcore_read_write(node);
            break;
        }
    }
    if (!ss.str().empty()) {
        logger->error("Code incompatibility detected");
        logger->error("Cannot translate mod file to .cpp file");
        logger->error("Fix the following errors and try again");
        std::string line;
        std::istringstream ss_stringstream(ss.str());
        while (std::getline(ss_stringstream, line)) {
            if (!line.empty())
                logger->error("Code Incompatibility :: {}"_format(line));
        }
        return true;
    }
    return false;
}

}  // namespace codegen
}  // namespace nmodl

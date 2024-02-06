/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/semantic_analysis_visitor.hpp"
#include "ast/breakpoint_block.hpp"
#include "ast/function_block.hpp"
#include "ast/function_call.hpp"
#include "ast/function_table_block.hpp"
#include "ast/independent_block.hpp"
#include "ast/procedure_block.hpp"
#include "ast/program.hpp"
#include "ast/statement_block.hpp"
#include "ast/string.hpp"
#include "ast/suffix.hpp"
#include "ast/table_statement.hpp"
#include "symtab/symbol_properties.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

bool SemanticAnalysisVisitor::check(const ast::Program& node) {
    check_fail = false;
    current_symbol_table = node.get_symbol_table();

    /// <-- This code is for check 2
    const auto& suffix_node = collect_nodes(node, {ast::AstNodeType::SUFFIX});
    if (!suffix_node.empty()) {
        const auto& suffix = std::dynamic_pointer_cast<const ast::Suffix>(suffix_node[0]);
        const auto& type = suffix->get_type()->get_node_name();
        is_point_process = (type == "POINT_PROCESS" || type == "ARTIFICIAL_CELL");
    }
    /// -->

    /// <-- This code is for check 4
    using namespace symtab::syminfo;
    const auto& with_prop = NmodlType::read_ion_var | NmodlType::write_ion_var;

    const auto& sym_table = node.get_symbol_table();
    assert(sym_table != nullptr);

    // this block is going nowhere since nodes below is just its declaration
    // and not all the nodes where it is used.
    auto property = NmodlType::random_var;
    auto random_variables = sym_table->get_variables_with_properties(property);
    for (const auto& var: random_variables) {
        auto nodes = var->get_nodes();
        for (auto node: nodes) {
            //            std::cout << "    " << node->get_node_type_name() << std::endl;
        }
    }

    // get all ion variables
    const auto& ion_variables = sym_table->get_variables_with_properties(with_prop, false);

    /// make sure ion variables aren't redefined in a `CONSTANT` block.
    for (const auto& var: ion_variables) {
        if (var->has_any_property(NmodlType::constant_var)) {
            logger->critical(
                fmt::format("SemanticAnalysisVisitor :: ion variable {} from the USEION statement "
                            "can not be re-declared in a CONSTANT block",
                            var->get_name()));
            check_fail = true;
        }
    }
    /// -->

    visit_program(node);
    return check_fail;
}

void SemanticAnalysisVisitor::visit_program(const ast::Program& node) {
    /// <-- This code is for check 8
    const auto& derivative_block_nodes = collect_nodes(node, {ast::AstNodeType::DERIVATIVE_BLOCK});
    if (derivative_block_nodes.size() > 1) {
        logger->critical("It is not supported to have several DERIVATIVE blocks");
        check_fail = true;
    }
    /// -->
    node.visit_children(*this);
}

void SemanticAnalysisVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    /// <-- This code is for check 1
    in_procedure = true;
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    in_procedure = false;
    /// -->
}

void SemanticAnalysisVisitor::visit_function_block(const ast::FunctionBlock& node) {
    /// <-- This code is for check 1
    in_function = true;
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    in_function = false;
    /// -->
}

void SemanticAnalysisVisitor::visit_statement_block(const ast::StatementBlock& node) {
    auto last_symbol_table = current_symbol_table;
    current_symbol_table = node.get_symbol_table();
    node.visit_children(*this);
    current_symbol_table = last_symbol_table;
}

void SemanticAnalysisVisitor::visit_name(const ast::Name& node) {
    /// <-- This code is a portion of  check 9
    // There are only two contexts where a random_var is allowed. As the first arg of a random
    // function or as an item in the RANDOM declaration.
    // Only the former needs checking.
    bool ok = true;
    if (node.is_name()) {  // E.g. for some reason SUFFIX arrives in visit_name
        auto name = node.get_node_name();
        auto symbol = current_symbol_table->lookup_in_scope(name);
        bool is_random_var =
            symbol ? (symbol->get_properties() == symtab::syminfo::NmodlType::random_var) : false;
        if (is_random_var) {
            auto parent = node.get_parent();
            if (parent) {
                if (parent->is_var_name()) {
                    parent = parent->get_parent();
                    if (parent && parent->is_function_call()) {
                        // The function must be a random function
                        auto fname = parent->get_node_name();
                        if (is_random_construct_function(fname)) {
                            // but is name the first arg?
                            ast::FunctionCall* rfun = (ast::FunctionCall*) parent;
                            const auto& arguments = rfun->get_arguments();
                            if (arguments[0]->get_node_name() != name) {
                                ok = false;
                            }
                        } else {
                            ok = false;
                        }
                    } else {
                        ok = false;
                    }
                } else {
                    assert(parent->is_random_var());
                }
            }
        }
    }
    if (!ok) {
        auto position = node.get_token()->position();  // ast::Name has a token
        logger->critical(
            fmt::format("SemanticAnalysisVisitor :: RANDOM variable {} at {}"
                        " can be used only as the first arg of a random function",
                        node.get_node_name(),
                        position));
        check_fail = true;
    }
    node.visit_children(*this);
    /// -->
}

void SemanticAnalysisVisitor::visit_function_call(const ast::FunctionCall& node) {
    /// <-- This code is a portion of  check 9
    //  The first arg of a random function must be a random_var
    auto fname = node.get_node_name();
    bool ok = true;
    if (is_random_construct_function(fname)) {
        const auto& arguments = node.get_arguments();
        if (arguments.empty()) {
            ok = false;
        } else {
            auto arg0 = arguments[0];
            if (arg0->is_var_name()) {
                auto name = arg0->get_node_name();
                auto symbol = current_symbol_table->lookup_in_scope(name);
                if (!symbol || symbol->get_properties() != symtab::syminfo::NmodlType::random_var) {
                    ok = false;
                }
            } else {
                ok = false;
            }
        }
    }
    if (!ok) {
        auto position = node.get_name()->get_token()->position();
        logger->critical(
            fmt::format("SemanticAnalysisVisitor :: random function {} at {} ::"
                        " The first arg must be a random variable",
                        fname,
                        position));
        check_fail = true;
    }
    node.visit_children(*this);
    /// -->
}

void SemanticAnalysisVisitor::visit_table_statement(const ast::TableStatement& tableStmt) {
    /// <-- This code is for check 1
    if ((in_function || in_procedure) && !one_arg_in_procedure_function) {
        logger->critical(
            "SemanticAnalysisVisitor :: The procedure or function containing the TABLE statement "
            "should contains exactly one argument.");
        check_fail = true;
    }
    /// -->
    /// <-- This code is for check 3
    const auto& table_vars = tableStmt.get_table_vars();
    if (in_function && !table_vars.empty()) {
        logger->critical(
            "SemanticAnalysisVisitor :: TABLE statement in FUNCTION cannot have a table name "
            "list.");
    }
    if (in_procedure && table_vars.empty()) {
        logger->critical(
            "SemanticAnalysisVisitor :: TABLE statement in PROCEDURE must have a table name list.");
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_destructor_block(const ast::DestructorBlock& /* node */) {
    /// <-- This code is for check 2
    if (!is_point_process) {
        logger->warn(
            "SemanticAnalysisVisitor :: This mod file is not point process but contains a "
            "destructor.");
        check_fail = true;
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_independent_block(const ast::IndependentBlock& node) {
    /// <-- This code is for check 5
    for (const auto& n: node.get_variables()) {
        if (n->get_value()->get_value() != "t") {
            logger->warn(
                "SemanticAnalysisVisitor :: '{}' cannot be used as an independent variable, only "
                "'t' is allowed.",
                n->get_value()->get_value());
        }
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_function_table_block(const ast::FunctionTableBlock& node) {
    /// <-- This code is for check 7
    if (node.get_parameters().size() < 1) {
        logger->critical(
            "SemanticAnalysisVisitor :: Function table '{}' must have one or more arguments.",
            node.get_node_name());
        check_fail = true;
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_protect_statement(const ast::ProtectStatement& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("PROTECT statement is not supported with GPU execution");
    }
    if (in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Find a PROTECT inside a already locked part.");
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_mutex_lock(const ast::MutexLock& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("MUTEXLOCK statement is not supported with GPU execution");
    }
    if (in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Found a MUTEXLOCK inside an already locked part.");
    }
    in_mutex = true;
    /// -->
}

void SemanticAnalysisVisitor::visit_mutex_unlock(const ast::MutexUnlock& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("MUTEXUNLOCK statement is not supported with GPU execution");
    }
    if (!in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Found a MUTEXUNLOCK outside a locked part.");
    }
    in_mutex = false;
    /// -->
}

}  // namespace visitor
}  // namespace nmodl

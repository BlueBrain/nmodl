/*
 * Copyright 2024 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/needsetdata_visitor.hpp"

#include <utility>

#include "ast/all.hpp"


namespace nmodl {
namespace visitor {

using symtab::Symbol;
using symtab::syminfo::NmodlType;

void NeedSetDataVisitor::visit_var_name(const ast::VarName& node) {
    if (function_or_procedure_stack.empty()) {
        return;
    }
    const auto var_sym = psymtab->lookup(node.get_node_name());
    const auto properties = NmodlType::range_var | NmodlType::pointer_var | NmodlType::bbcore_pointer_var;
    if (var_sym && var_sym->has_any_property(properties)) {
        const auto is_function = function_or_procedure_stack.top()->is_function_block();
        const auto func_or_proc_str = is_function ? "Function" : "Procedure";
        std::cout << func_or_proc_str << " " << function_or_procedure_stack.top()->get_node_name() << " has range var: " << node.get_node_name() << std::endl;
    }
    function_proc_need_setdata.insert(function_or_procedure_stack.top());
    auto func_symbol = psymtab->lookup(function_or_procedure_stack.top()->get_node_name());
    std::cout << "1Adding NmodlType::need_setdata to " << func_symbol->get_name() << std::endl;
    func_symbol->add_property(NmodlType::need_setdata);
}

void NeedSetDataVisitor::visit_function_call(const ast::FunctionCall& node) {
    std::cout << "Calling " << node.get_node_name() << std::endl;
    auto func_symbol = psymtab->lookup(node.get_node_name());
    const auto func_block = func_symbol->get_nodes()[0];
    func_block->accept(*this);
    if (function_proc_need_setdata.find(dynamic_cast<const ast::Block*>(func_block)) != function_proc_need_setdata.end()) {
        std::cout << "Adding to the set " << node.get_node_name() << std::endl;
        function_proc_need_setdata.insert(function_or_procedure_stack.top());
    }
    auto caller_func_symbol = psymtab->lookup(function_or_procedure_stack.top()->get_node_name());
    std::cout << "2Adding NmodlType::need_setdata to " << caller_func_symbol->get_name() << std::endl;
    caller_func_symbol->add_property(NmodlType::need_setdata);
    std::cout << "Leaving " << node.get_node_name() << std::endl;
}

void NeedSetDataVisitor::visit_function_block(const ast::FunctionBlock& node) {
    function_or_procedure_stack.push(&node);
    std::cout << "In " << node.get_node_name() << std::endl;
    node.visit_children(*this);
    std::cout << "Leaving " << node.get_node_name() << std::endl;
    function_or_procedure_stack.pop();
}

void NeedSetDataVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    function_or_procedure_stack.push(&node);
    std::cout << "In " << node.get_node_name() << std::endl;
    node.visit_children(*this);
    std::cout << "Leaving " << node.get_node_name() << std::endl;
    function_or_procedure_stack.pop();
}

void NeedSetDataVisitor::visit_program(const ast::Program& node) {
    psymtab = node.get_symbol_table();
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl

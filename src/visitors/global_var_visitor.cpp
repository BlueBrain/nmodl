/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>

#include "ast/ast.hpp"
#include "global_var_visitor.hpp"
/**
 * \file
 * \brief AST Visitor to parse the ast::UnitDefs and ast::FactorDefs from the mod file
 * by the Units Parser used to parse the \c nrnunits.lib file
 */

namespace nmodl {
namespace visitor {

using symtab::syminfo::NmodlType;

/**
 * \details units::Unit definition is based only on pre-defined units, parse only the
 * new unit and the pre-defined units. <br>
 * Example:
 * \code
 *      (nA)    = (nanoamp) => nA  nanoamp)
 * \endcode
 * The ast::UnitDef is converted to a string that is able to be parsed by the
 * unit parser which was used for parsing the \c nrnunits.lib file.
 * On \c nrnunits.lib constant "1" is defined as "fuzz", so it must be converted.
 */
void GlobalToRangeVisitor::visit_global_var(ast::GlobalVar* node) {
    std::cout << node->get_node_name() << std::endl;
    if (ast->get_symbol_table()->lookup(node->get_node_name())->get_write_count() > 0) {
        ast->get_symbol_table()
            ->lookup(node->get_node_name())
            ->remove_property(NmodlType::global_var);
        ast->get_symbol_table()->lookup(node->get_node_name())->add_property(NmodlType::range_var);
        std::cout << ast->get_symbol_table()->lookup(node->get_node_name())->get_properties()
                  << std::endl;
    }
}

void GlobalToRangeVisitor::visit_neuron_block(ast::NeuronBlock* node) {
    auto symtab = node->get_symbol_table();
    auto global_vars = symtab->get_variables_with_properties(NmodlType::global_var, false);
    for (auto global_var: global_vars) {
        if (ast->get_symbol_table()->lookup(global_var->get_name())->get_write_count() > 0) {
            symtab->lookup(global_var->get_name())->remove_property(NmodlType::global_var);
            symtab->lookup(global_var->get_name())->add_property(NmodlType::range_var);
            std::cout << symtab->lookup(global_var->get_name())->get_properties() << std::endl;
        }
    }
}

void GlobalToRangeVisitor::visit_range_var(ast::RangeVar* node) {
    std::cout << node->get_node_name() << std::endl;
    std::cout << ast->get_symbol_table()->lookup(node->get_node_name())->get_properties()
              << std::endl;
}

}  // namespace visitor
}  // namespace nmodl

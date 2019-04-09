/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/units_visitor.hpp"
#include "ast/ast.hpp"
#include "visitors/lookup_visitor.hpp"
#include "visitors/visitor_utils.hpp"

#include <iostream>
#include <memory>

namespace nmodl {

void UnitsVisitor::visit_unit_def(ast::UnitDef* node) {
    std::stringstream ss;
    // Unit definition is based only on pre-defined units, parse only the new unit
    // and the pre-defined units (ex. (nA)    = (nanoamp) => nA  nanoamp)
    // On nrnunits.lib constant "1" is defined as "fuzz"
    if (node->get_unit2()->get_node_name() == "1") {
        ss << node->get_unit1()->get_node_name() << "\t"
           << "fuzz";
    } else {
        ss << node->get_unit1()->get_node_name() << "\t" << node->get_unit2()->get_node_name();
    }
    if (verbose) {
        std::cout << to_nmodl(node) << std::endl;
        std::cout << ss.str() << std::endl;
    }
    unit_driver.parse_string(ss.str());
}

void UnitsVisitor::visit_factor_def(ast::FactorDef* node) {
    std::stringstream ss;

    // The new unit definition is based on the factor of other defined units or a factor which is a
    // number, parse only the new unit with the units that it's based on in the first case
    // (ex. FARADAY = (faraday) (coulomb) => FARADAY faraday) or parse the new unit, with
    // its factor and the units that it's based on in the second case   (ex. dummy   =
    // 123.45  (m/sec2) => dummy    123.45 m/sec2)
    if (node->get_value() != NULL) {
        ss << node->get_node_name() << "\t" << node->get_value()->get_value() << " "
           << node->get_unit1()->get_node_name();
    } else {
        ss << node->get_node_name() << "\t" << node->get_unit1()->get_node_name();
    }
    if (verbose) {
        std::cout << to_nmodl(node) << std::endl;
        std::cout << ss.str() << std::endl;
    }
    unit_driver.parse_string(ss.str());

    // Save factor of unit calculated in the UnitsTable to the AST node
    auto node_unit_name = node->get_node_name();
    auto unit_factor = unit_driver.Table->get_unit(node_unit_name)->get_factor();
    auto double_value_ptr = std::make_shared<ast::Double>(ast::Double(unit_factor));
    node->set_value(static_cast<std::shared_ptr<ast::Double>&&>(double_value_ptr));
}

}  // namespace nmodl
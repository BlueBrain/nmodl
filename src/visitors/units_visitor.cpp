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

void UnitsVisitor::visit_program(ast::Program* node) {
    unit_driver.parse_file(units_dir);
    node->visit_children(this);
}

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

    unit_driver.parse_string(ss.str());

    if (verbose) {
        auto unit = unit_driver.table->get_unit(node->get_node_name());
        *units_details << std::fixed << std::setprecision(8) << unit->get_name() << " "
                       << unit->get_factor() << ":";
        for (const auto& dims: unit->get_dims()) {
            *units_details << " " << dims;
        }
        *units_details << "\n";
    }
}

void UnitsVisitor::visit_factor_def(ast::FactorDef* node) {
    std::stringstream ss;
    std::cout << node->get_node_name() << std::endl;
    // The new unit definition is based on the factor of other defined units or a factor which is a
    // number
    // Parse only the new unit with the units that it's based on in the first case
    // (ex. FARADAY = (faraday) (coulomb) => FARADAY faraday)
    // Parse the new unit, with its factor and the units that it's based on in the second case
    // (ex. dummy   = 123.45  (m/sec2) => dummy    123.45 m/sec2)
    std::string node1_name;
    if (node->get_unit1()->get_node_name() == "1") {
        node1_name = "fuzz";
    } else {
        node1_name = node->get_unit1()->get_node_name();
    }
    if (node->get_value() != NULL) {
        ss << node->get_node_name() << "\t" << node->get_value()->get_value() << " " << node1_name;
    } else {
        ss << node->get_node_name() << "\t" << node1_name;
    }

    unit_driver.parse_string(ss.str());

    // Save factor of unit calculated in the UnitsTable to the AST node if it was not defined in the
    // mod file
    if (node->get_value() == NULL) {
        auto node_unit_name = node->get_node_name();
        auto unit_factor = unit_driver.table->get_unit(node_unit_name)->get_factor();
        auto double_value_ptr = std::make_shared<ast::Double>(ast::Double(unit_factor));
        node->set_value(static_cast<std::shared_ptr<ast::Double>&&>(double_value_ptr));
    }
    if (verbose) {
        auto unit = unit_driver.table->get_unit(node->get_node_name());
        // print value of unit that will be printed to the .cpp file and not the value that is
        // calculated based on the UnitTable units value
        *units_details << std::fixed << std::setprecision(8) << unit->get_name() << " "
                       << node->get_value()->get_value() << ":";
        for (const auto& dims: unit->get_dims()) {
            *units_details << " " << dims;
        }
        *units_details << "\n";
    }
}

}  // namespace nmodl
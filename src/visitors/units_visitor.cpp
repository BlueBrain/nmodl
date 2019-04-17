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
    // The UnitDef is converted to a string that is able to be parsed by the
    // unit parser which was used for parsing the nrnunits.lib file
    // On nrnunits.lib constant "1" is defined as "fuzz", so it must be converted
    if (node->get_unit2()->get_node_name() == "1") {
        ss << node->get_unit1()->get_node_name() << "\t"
           << "fuzz";
    } else {
        ss << node->get_unit1()->get_node_name() << "\t" << node->get_unit2()->get_node_name();
    }

    unit_driver.parse_string(ss.str());

    if (verbose) {
        auto unit_name = node->get_node_name();
        unit_name.erase(remove_if(unit_name.begin(), unit_name.end(), isspace), unit_name.end());
        auto unit = unit_driver.table->get_unit(unit_name);
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
    // The new unit definition is based on a factor combined with units or
    // other defined units.
    // In the first case the factor saved to the AST node and printed to
    // .cpp file is the one defined on the modfile. The factor and the
    // dimensions saved to the UnitTable are based on the factor and the
    // units defined in the modfile, so this factor will be calculated
    // based on the base units of the UnitTable.
    // In the second case, the factor and the dimensions that are inserted
    // to the UniTable are based on the unit1 of the FactorDef, like MOD2C.
    // However, the factor that is saved in the AST node and printed in the
    // .cpp file is the factor of the unit1 devided by the factor of unit2.
    // To parse the units defined in modfiles there are stringstreams
    // created that are passed to the string parser, to be parsed by the
    // unit parser used for parsing the nrnunits.lib file, which takes care
    // of all the units calculations.
    auto node_has_value_defined_in_modfile = node->get_value() != NULL;
    if (node_has_value_defined_in_modfile) {
        // In nrnunits.lib file "1" is defined as "fuzz", so there must
        // be a conversion to be able to to parse "1" as unit
        if (node->get_unit1()->get_node_name() == "1") {
            ss << node->get_node_name() << "\t" << node->get_value()->get_value() << " fuzz";
        } else {
            ss << node->get_node_name() << "\t" << node->get_value()->get_value() << " "
               << node->get_unit1()->get_node_name();
        }
    } else {
        std::stringstream ss_unit1, ss_unit2;
        std::string unit1_name, unit2_name;
        if (node->get_unit1()->get_node_name() == "1") {
            unit1_name = "fuzz";
        } else {
            unit1_name = node->get_unit1()->get_node_name();
        };
        if (node->get_unit2()->get_node_name() == "1") {
            unit2_name = "fuzz";
        } else {
            unit2_name = node->get_unit2()->get_node_name();
        };
        ss_unit1 << node->get_node_name() << "_unit1\t" << unit1_name;
        unit_driver.parse_string(ss_unit1.str());
        ss_unit2 << node->get_node_name() << "_unit2\t" << unit2_name;
        unit_driver.parse_string(ss_unit2.str());
        ss << node->get_node_name() << "\t" << unit1_name;
    }

    unit_driver.parse_string(ss.str());

    // If the FactorDef was done by using 2 units, there must be calculated
    // the factors of both of them based on the UnitTable and then they must
    // be divided to produce the unit's factor that will be printed to the
    // .cpp file.
    if (!node_has_value_defined_in_modfile) {
        auto node_unit_name = node->get_node_name();
        auto unit1_factor = unit_driver.table->get_unit(node_unit_name + "_unit1")->get_factor();
        auto unit2_factor = unit_driver.table->get_unit(node_unit_name + "_unit2")->get_factor();
        auto unit_factor = unit1_factor / unit2_factor;
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

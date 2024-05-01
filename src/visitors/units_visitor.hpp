/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::UnitsVisitor
 */

#include <string>

#include "parser/unit_driver.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class UnitsVisitor
 * \brief Visitor for Units blocks of AST
 *
 * This is simple example of visitor that uses base AstVisitor
 * interface. We override AstVisitor::visit_program, AstVisitor::visit_unit_def
 * and AstVisitor::visit_factor_def method. Furthermore it keeps the
 * parser::UnitDriver to parse the units file and the strings generated by the
 * units in the mod files.
 */

class UnitsVisitor: public AstVisitor {
  private:
    /// Units Driver needed to parse the units file and the string produces by
    /// mod files' units
    parser::UnitDriver units_driver;

    /// Directory of units lib file that defines all the basic units
    std::string units_dir;

    /// Declaration of `fuzz` constant unit, which is the equivilant of `1`
    /// in mod files UNITS definitions
    const std::string UNIT_FUZZ = "fuzz";

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default UnitsVisitor constructor
    UnitsVisitor() = default;

    /// UnitsVisitor constructor that takes as argument the units file to parse
    /// the units from
    explicit UnitsVisitor(std::string t_units_dir)
        : units_dir(std::move(t_units_dir)) {}

    /// \}

    /// Function to visit all the ast::UnitDef nodes and parse the units defined as
    /// ast::UnitDef in the UNITS block of mod files
    void visit_unit_def(ast::UnitDef& node) override;

    /// Function to visit all the ast::FactorDef nodes and parse the units defined
    /// as ast::FactorDef in the UNITS block of mod files
    void visit_factor_def(ast::FactorDef& node) override;

    /// Override visit_program function to parse the \c nrnunits.lib unit file
    /// before starting visiting the AST to parse the units defined in mod files
    void visit_program(ast::Program& node) override;

    /// Get the parser::UnitDriver to be able to use it outside the visitor::UnitsVisitor
    /// scope keeping the same units::UnitTable
    const parser::UnitDriver& get_unit_driver() const noexcept {
        return units_driver;
    }
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fstream>

#include "CLI/CLI.hpp"
#include "fmt/format.h"

#include "config/config.h"
#include "nmodl_driver.hpp"
#include "parser/unit_driver.hpp"
#include "unit_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/units_visitor.hpp"

/**
 * Standalone parser program for Units block of mod files. This demonstrate basic
 * usage of parser and driver class for parsing UNITS block of mod files.
 */

using namespace fmt::literals;
using namespace nmodl;
using namespace visitor;

void parse_units(const std::vector<std::string>& units_files,
                 const std::vector<std::string>& mod_files) {
    for (const auto& mod_f: mod_files) {
        logger->info("Processing {}", mod_f);
        std::ifstream file(mod_f);

        /// driver object creates lexer and parser
        parser::NmodlDriver driver;
        driver.set_verbose(false);

        /// parse mod file to create AST
        driver.parse_file(mod_f);

        auto ast = driver.get_ast();

        /// visit AST nodes to parse the defined Units in the UNITS blocks
        /// of the mod file
        for (auto unit_f: units_files) {
            logger->info("Parsing UNITS using unit file {}", unit_f);
            std::stringstream ss;
            UnitsVisitor(unit_f, ss).visit_program(ast.get());
            std::cout << ss.str();
        }
    }
}

int main(int argc, const char* argv[]) {
    CLI::App app{"UNITS-Parser : Standalone Parser for UNITS block of mod files({})"_format(
        Version::to_string())};

    std::vector<std::string> unit_files, mod_files;
    unit_files.push_back(NrnUnitsLib::get_path());
    app.add_option("mod_files,--mod_files", mod_files, "One or more mod files to process")
        ->ignore_case()
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--unit_files", unit_files, "One or more Units files to process", true);

    CLI11_PARSE(app, argc, argv);

    parse_units(unit_files, mod_files);

    return 0;
}

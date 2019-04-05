/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fstream>

#include "CLI/CLI.hpp"
#include "fmt/format.h"

#include "parser/unit_driver.hpp"
#include "utils/logger.hpp"
#include "version/version.h"

/**
 * Standalone parser program for Units. This demonstrate basic
 * usage of parser and driver class.
 */

using namespace fmt::literals;
using namespace nmodl;

int main(int argc, const char* argv[]) {
    CLI::App app{"Unit-Parser : Standalone Parser for Units({})"_format(version::to_string())};

    std::vector<std::string> files;
    app.add_option("file", files, "One or more Units files to process")
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    for (const auto& f: files) {
        logger->info("Processing {}", f);
        std::ifstream file(f);

        /// driver object creates lexer and parser
        parser::UnitDriver driver;
        driver.set_verbose(true);

        /// just call parser method
        driver.parse_stream(file);

        /// Print all tokens read
        // driver.print_tokens();

        /// Print Units and their factors
        std::cout << " -- PRINTING TABLE -- " << std::endl;
        driver.Table->print_units();

        std::cout << " -- PRINTING BASE UNITS -- " << std::endl;
        driver.Table->print_base_units();
    }

    return 0;
}

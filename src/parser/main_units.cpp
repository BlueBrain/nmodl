/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>

#include <CLI/CLI.hpp>

#include "config/config.h"
#include "parser/unit_driver.hpp"
#include "utils/logger.hpp"

/**
 * Standalone parser program for Units. This demonstrate basic
 * usage of parser and driver class to parse the `nrnunits.lib`
 * file.
 *
 */

using namespace nmodl;


int main(int argc, const char* argv[]) {
    CLI::App app{
        fmt::format("Unit-Parser : Standalone Parser for Units({})", Version::to_string())};

    std::vector<std::string> units_files;
    units_files.push_back(NrnUnitsLib::get_path());
    app.add_option("units_files", units_files, "One or more Units files to process");

    CLI11_PARSE(app, argc, argv);

    for (const auto& f: units_files) {
        logger->info("Processing {}", f);
        std::ifstream file(f);

        // driver object creates lexer and parser
        parser::UnitDriver driver;
        driver.set_verbose(true);

        // just call parser method
        driver.parse_stream(file);
        driver.table->print_units_sorted(std::cout);
    }

    return 0;
}

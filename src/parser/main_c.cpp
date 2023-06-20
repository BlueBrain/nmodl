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

#include <CLI/CLI.hpp>

#include "config/config.h"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"

/**
 * \file
 * Standalone parser program for C. This demonstrate basic
 * usage of parser and driver class.
 */

using namespace nmodl;

int main(int argc, const char* argv[]) {
    CLI::App app{fmt::format("C-Parser : Standalone Parser for C Code({})", Version::to_string())};

    std::vector<std::string> files;
    app.add_option("file", files, "One or more C files to process")
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    for (const auto& f: files) {
        logger->info("Processing {}", f);
        std::ifstream file(f);

        /// driver object creates lexer and parser
        parser::CDriver driver;
        driver.set_verbose(true);

        /// just call parser method
        driver.parse_stream(file);
    }
    return 0;
}

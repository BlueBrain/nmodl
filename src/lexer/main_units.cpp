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

#include "config/config.h"
#include "lexer/unit_lexer.hpp"
#include "parser/unit_driver.hpp"
#include "utils/logger.hpp"

#include <CLI/CLI.hpp>

#include <fstream>

/**
 * \file
 * Example of standalone lexer program for Units that
 * demonstrate use of UnitLexer and UnitDriver classes.
 */

using namespace nmodl;
using Token = parser::UnitParser::token;

int main(int argc, const char* argv[]) {
    CLI::App app{fmt::format("Unit-Lexer : Standalone Lexer for Units({})", Version::to_string())};

    std::vector<std::string> files;
    app.add_option("file", files, "One or more units files to process")
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    for (const auto& f: files) {
        nmodl::logger->info("Processing {}", f);
        std::ifstream file(f);
        nmodl::parser::UnitDriver driver;
        nmodl::parser::UnitLexer scanner(driver, &file);

        /// parse Units file and print token until EOF
        while (true) {
            auto sym = scanner.next_token();
            auto token_type = sym.type_get();
            if (token_type == parser::UnitParser::by_type(Token::END).type_get()) {
                break;
            }
            std::cout << sym.value.as<std::string>() << std::endl;
        }
    }
    return 0;
}

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
#include <sstream>

#include "lexer/unit_lexer.hpp"
#include "parser/unit_driver.hpp"

namespace nmodl {
namespace parser {

/// parse Units file provided as istream
bool UnitDriver::parse_stream(std::istream& in) {
    UnitLexer scanner(*this, &in);
    UnitParser parser(scanner, *this);

    this->lexer = &scanner;
    this->parser = &parser;

    return (parser.parse() == 0);
}

/// parse Units file
bool UnitDriver::parse_file(const std::string& filename) {
    std::ifstream in(filename.c_str());
    stream_name = filename;

    if (!in.good()) {
        return false;
    }
    return parse_stream(in);
}

/// parser Units provided as string (used for testing)
bool UnitDriver::parse_string(const std::string& input) {
    std::istringstream iss(input);
    return parse_stream(iss);
}

}  // namespace parser
}  // namespace nmodl

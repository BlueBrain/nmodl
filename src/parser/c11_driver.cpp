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

#include "lexer/c11_lexer.hpp"
#include "parser/c11_driver.hpp"

namespace nmodl {
namespace parser {

CDriver::CDriver() = default;

CDriver::CDriver(bool strace, bool ptrace)
    : trace_scanner(strace)
    , trace_parser(ptrace) {}

CDriver::~CDriver() = default;

/// parse c file provided as istream
bool CDriver::parse_stream(std::istream& in) {
    lexer.reset(new CLexer(*this, &in));
    parser.reset(new CParser(*lexer, *this));

    lexer->set_debug(trace_scanner);
    parser->set_debug_level(trace_parser);
    return parser->parse() == 0;
}

//// parse c file
bool CDriver::parse_file(const std::string& filename) {
    std::ifstream in(filename.c_str());
    streamname = filename;

    if (!in.good()) {
        return false;
    }
    return parse_stream(in);
}

/// parser c provided as string (used for testing)
bool CDriver::parse_string(const std::string& input) {
    std::istringstream iss(input);
    return parse_stream(iss);
}

void CDriver::error(const std::string& m) {
    std::cerr << m << '\n';
}

void CDriver::add_token(const std::string& text) {
    tokens.push_back(text);
    // here we will query and look into symbol table or register callback
}

void CDriver::error(const std::string& m, const location& l) {
    std::cerr << l << " : " << m << '\n';
}

void CDriver::scan_string(const std::string& text) {
    std::istringstream in(text);
    lexer.reset(new CLexer(*this, &in));
    parser.reset(new CParser(*lexer, *this));
    while (true) {
        auto sym = lexer->next_token();
        auto token_type = sym.type_get();
        if (token_type == CParser::by_type(CParser::token::END).type_get()) {
            break;
        }
    }
}

}  // namespace parser
}  // namespace nmodl

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

#include <sstream>
#include <utility>

#include "lexer/diffeq_lexer.hpp"
#include "parser/diffeq_driver.hpp"
#include "utils/string_utils.hpp"

namespace nmodl {
namespace parser {

void DiffeqDriver::parse_equation(const std::string& equation,
                                  std::string& state,
                                  std::string& rhs,
                                  int& order) {
    auto parts = stringutils::split_string(equation, '=');
    state = stringutils::trim(parts[0]);
    rhs = stringutils::trim(parts[1]);

    /// expect prime on lhs, find order and remove quote
    auto const wide_order = std::count(state.begin(), state.end(), '\'');
    assert(wide_order >= 0 && wide_order <= std::numeric_limits<int>::max());
    order = static_cast<int>(wide_order);
    stringutils::remove_character(state, '\'');

    /// error if no prime in equation or not an assignment statement
    if (order == 0 || state.empty()) {
        throw std::runtime_error("Invalid equation, no prime on rhs? " + equation);
    }
}

std::string DiffeqDriver::solve(const std::string& equation, std::string method, bool debug) {
    std::string state, rhs;
    int order = 0;
    bool cnexp_possible{};
    parse_equation(equation, state, rhs, order);
    return solve_equation(state, order, rhs, method, cnexp_possible, debug);
}

std::string DiffeqDriver::solve_equation(std::string& state,
                                         int order,
                                         std::string& rhs,
                                         std::string& method,
                                         bool& cnexp_possible,
                                         bool debug) {
    std::istringstream in(rhs);
    diffeq::DiffEqContext eq_context(state, order, rhs, method);
    DiffeqLexer scanner(&in);
    DiffeqParser parser(scanner, eq_context);
    parser.parse();
    if (debug) {
        eq_context.print();
    }
    return eq_context.get_solution(cnexp_possible);
}

/// \todo Instead of using neuron like api, we need to refactor
bool DiffeqDriver::cnexp_possible(const std::string& equation, std::string& solution) {
    std::string state, rhs;
    int order = 0;
    bool cnexp_possible{};
    std::string method = "cnexp";
    parse_equation(equation, state, rhs, order);
    solution = solve_equation(state, order, rhs, method, cnexp_possible);
    return cnexp_possible;
}

}  // namespace parser
}  // namespace nmodl

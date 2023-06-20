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

#pragma once

/**
 * \file token_mapping.hpp
 * \brief Map different tokens from lexer to token types
 */

#include <string>

#include "parser/nmodl/nmodl_parser.hpp"

namespace nmodl {

bool is_keyword(const std::string& name);
bool is_method(const std::string& name);

parser::NmodlParser::token_type token_type(const std::string& name);
std::vector<std::string> get_external_variables();
std::vector<std::string> get_external_functions();

namespace details {

bool needs_neuron_thread_first_arg(const std::string& token);

}  // namespace details

}  // namespace nmodl

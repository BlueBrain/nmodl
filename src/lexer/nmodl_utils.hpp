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
 * \file nmodl_utils.hpp
 * \brief Utility functions for NMODL lexer
 *
 * From nmodl lexer we return different symbols to parser. Instead of writing
 * those functions in the flex implementation file, those commonly used routines
 * are defined here. Some of these tasks were implemented in list.c file in the
 * original mod2c implementation.
 */

#include "parser/nmodl/location.hh"
#include "parser/nmodl/nmodl_parser.hpp"

namespace nmodl {

using PositionType = parser::location;
using SymbolType = parser::NmodlParser::symbol_type;
using Token = parser::NmodlParser::token;
using TokenType = parser::NmodlParser::token_type;

SymbolType double_symbol(const std::string& value, PositionType& pos);
SymbolType integer_symbol(int value, PositionType& pos, const char* text = nullptr);
SymbolType name_symbol(const std::string& text, PositionType& pos, TokenType type = Token::NAME);
SymbolType prime_symbol(std::string text, PositionType& pos);
SymbolType string_symbol(const std::string& text, PositionType& pos);
SymbolType token_symbol(const std::string& key, PositionType& pos, TokenType type = Token::UNKNOWN);

}  // namespace nmodl

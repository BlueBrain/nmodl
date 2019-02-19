/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "parser/nmodl/location.hh"
#include "parser/nmodl/nmodl_parser.hpp"

/**
 * \brief Utility functions for nmodl lexer
 *
 * From nmodl lexer we return different symbols to parser.
 * Instead of writing those functions in the flex implementation
 * file, those commonly used routines are defined here. Some of
 * these tasks were implemented in list.c file in the oiginal mod2c
 * implementation.
 */

namespace nmodl {
using PositionType = nmodl::location;
using SymbolType = nmodl::Parser::symbol_type;
using Token = nmodl::Parser::token;
using TokenType = nmodl::Parser::token_type;

SymbolType double_symbol(double value, PositionType& pos);
SymbolType integer_symbol(int value, PositionType& pos, const char* text = nullptr);
SymbolType name_symbol(const std::string& text, PositionType& pos, TokenType type = Token::NAME);
SymbolType prime_symbol(std::string text, PositionType& pos);
SymbolType string_symbol(const std::string& text, PositionType& pos);
SymbolType token_symbol(const std::string& key, PositionType& pos, TokenType type = Token::UNKNOWN);

}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include <string>

#include "catch/catch.hpp"
#include "lexer/modtoken.hpp"
#include "lexer/nmodl_lexer.hpp"
#include "parser/nmodl_driver.hpp"

using Token = nmodl::Parser::token;

/// just retrieve token type from lexer
nmodl::Parser::token_type token_type(const std::string& name) {
    std::istringstream ss(name);
    std::istream& in = ss;

    nmodl::Driver driver;
    nmodl::Lexer scanner(driver, &in);

    using TokenType = nmodl::Parser::token_type;
    using SymbolType = nmodl::Parser::symbol_type;

    SymbolType sym = scanner.next_token();
    TokenType token = sym.token();

    /** Lexer returns raw pointers for some AST types
     * and we need to clean-up memory for those.
     * Todo: add tests later for checking values */
    switch (token) {
    case Token::NAME:
    case Token::METHOD:
    case Token::SUFFIX:
    case Token::VALENCE:
    case Token::DEL:
    case Token::DEL2: {
        auto value = sym.value.as<ast::Name>();
        break;
    }

    case Token::PRIME: {
        auto value = sym.value.as<ast::PrimeName>();
        break;
    }

    case Token::INTEGER: {
        auto value = sym.value.as<ast::Integer>();
        break;
    }

    case Token::REAL: {
        auto value = sym.value.as<ast::Double>();
        break;
    }

    case Token::STRING: {
        auto value = sym.value.as<ast::String>();
        break;
    }

    case Token::VERBATIM:
    case Token::BLOCK_COMMENT:
    case Token::LINE_PART: {
        auto value = sym.value.as<std::string>();
        break;
    }

    default: { auto value = sym.value.as<ModToken>(); }
    }

    return token;
}

TEST_CASE("Lexer tests for valid tokens", "[Lexer]") {
    SECTION("Tests for some keywords") {
        REQUIRE(token_type("VERBATIM Hello ENDVERBATIM") == Token::VERBATIM);
        REQUIRE(token_type("INITIAL") == Token::INITIAL1);
        REQUIRE(token_type("SOLVE") == Token::SOLVE);
    }

    SECTION("Tests for language constructs") {
        REQUIRE(token_type(" h' = (hInf-h)/hTau\n") == Token::PRIME);
        REQUIRE(token_type("while") == Token::WHILE);
        REQUIRE(token_type("if") == Token::IF);
        REQUIRE(token_type("else") == Token::ELSE);
        REQUIRE(token_type("WHILE") == Token::WHILE);
        REQUIRE(token_type("IF") == Token::IF);
        REQUIRE(token_type("ELSE") == Token::ELSE);
    }

    SECTION("Tests for valid numbers") {
        REQUIRE(token_type("123") == Token::INTEGER);
        REQUIRE(token_type("123.32") == Token::REAL);
        REQUIRE(token_type("1.32E+3") == Token::REAL);
        REQUIRE(token_type("1.32e-3") == Token::REAL);
        REQUIRE(token_type("32e-3") == Token::REAL);
    }

    SECTION("Tests for Name/Strings") {
        REQUIRE(token_type("neuron") == Token::NAME);
        REQUIRE(token_type("\"Quoted String\"") == Token::STRING);
    }

    SECTION("Tests for (math) operators") {
        REQUIRE(token_type(">") == Token::GT);
        REQUIRE(token_type(">=") == Token::GE);
        REQUIRE(token_type("<") == Token::LT);
        REQUIRE(token_type("==") == Token::EQ);
        REQUIRE(token_type("!=") == Token::NE);
        REQUIRE(token_type("<->") == Token::REACT1);
        REQUIRE(token_type("~+") == Token::NONLIN1);
        // REQUIRE( token_type("~") == Token::REACTION);
    }

    SECTION("Tests for braces") {
        REQUIRE(token_type("{") == Token::OPEN_BRACE);
        REQUIRE(token_type("}") == Token::CLOSE_BRACE);
        REQUIRE(token_type("(") == Token::OPEN_PARENTHESIS);
        REQUIRE(token_type(")") != Token::OPEN_PARENTHESIS);
    }
}

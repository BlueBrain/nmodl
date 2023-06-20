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

#include "parser/unit/unit_parser.hpp"

/**
 * Flex expects the declaration of yylex to be defined in the macro YY_DECL
 * and Unit parser class expects it to be declared.
 */
#ifndef YY_DECL
#define YY_DECL nmodl::parser::UnitParser::symbol_type nmodl::parser::UnitLexer::next_token()
#endif

/**
 * For creating multiple (different) lexer classes, we can use `-P` flag
 * (or prefix option) to rename each yyFlexLexer to some other name like
 * `xxFlexLexer`. And then include <FlexLexer.h> in other sources once per
 * lexer class, first renaming `yyFlexLexer` as shown below.
 */
#ifndef __FLEX_LEXER_H
#define yyFlexLexer UnitFlexLexer
#include "FlexLexer.h"
#endif

namespace nmodl {
namespace parser {

/**
 * @addtogroup lexer
 * @addtogroup units
 * @{
 */

/**
 * \class UnitLexer
 * \brief Represent Lexer/Scanner class for Units parsing
 *
 * Lexer defined to add some extra function to the scanner class from flex.
 * Flex itself creates yyFlexLexer class, which we renamed using macros to
 * UnitFlexLexer. But we change the context of the generated yylex() function
 * because the yylex() defined in UnitFlexLexer has no parameters.
 */
class UnitLexer: public UnitFlexLexer {
  public:
    /// location of the parsed token
    location loc;

    /// \name Ctor & dtor
    /// \{

    /**
     * \brief UnitLexer constructor
     *
     * @param driver UnitDriver where this lexer resides
     * @param in Input stream from where tokens will be read
     * @param out Output stream where output will be sent
     */
    explicit UnitLexer(UnitDriver& /* driver */,
                       std::istream* in = nullptr,
                       std::ostream* out = nullptr)
        : UnitFlexLexer(in, out) {}

    ~UnitLexer() override = default;

    /// \}

    /**
     * \brief Function for lexer to scan token (replacement for \c yylex())
     *
     * This is main lexing function generated by `flex` according to the macro
     * declaration \c YY_DECL. The generated bison parser then calls this virtual
     * function to fetch new tokens. Note that \c yylex() has different declaration
     * and hence can't be used for new lexer.
     *
     * @return Symbol encapsulating parsed token
     */
    virtual UnitParser::symbol_type next_token();
};

}  // namespace parser
}  // namespace nmodl

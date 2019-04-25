/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "ast/ast.hpp"
#include "parser/nmodl/nmodl_parser.hpp"


/**
 * Flex expects the declaration of yylex() to be defined in the macro YY_DECL
 * and C++ parser class expects it to be declared.
 */
#ifndef YY_DECL
#define YY_DECL nmodl::parser::NmodlParser::symbol_type nmodl::parser::NmodlLexer::next_token()
#endif

/**
 * For creating multiple (different) lexer classes, we can use `-P` flag
 * (or prefix option) to rename each `NmodlFlexLexer` to some other name like
 * `xxFlexLexer`. And then include <FlexLexer.h> in other sources once per
 * lexer class, first renaming `yyFlexLexer` as shown below.
 */
#ifndef __FLEX_LEXER_H
#define yyFlexLexer NmodlFlexLexer
#include "FlexLexer.h"
#endif

namespace nmodl {
namespace parser {

/**
 * @defgroup lexer Lexer Implementation
 * @brief All lexer classes implementation
 *
 * @addtogroup lexer
 * @{
 */

/**
 * \class NmodlLexer
 * \brief Represent Lexer/Scanner class for NMODL language parsing
 *
 * Lexer defined to add some extra function to the scanner class from flex.
 * Flex itself creates yyFlexLexer class, which we renamed using macros to
 * NmodlFlexLexer. But we change the context of the generated yylex() function
 * because the yylex() defined in NmodlFlexLexer has no parameters. Also, note
 * that implementation of the member functions are in nmodl.l file due to use
 * of macros.
 */
class NmodlLexer: public NmodlFlexLexer {
    /**
     * \brief Reference to driver object where this lexer resides
     *
     * The driver object is used for macro definitions and error checking
     */
    NmodlDriver& driver;

    /// Units are stored in the scanner (could be stored in the driver though)
    ast::String* last_unit = nullptr;

    /**
     * \brief Context of the reaction (`~`) token
     *
     * For reaction (`~`) we return different token based on one of the following
     * lexical context:
     * - NONLINEAR
     * - LINEAR
     * - KINETIC
     * - PARTIAL
     */
    int lexical_context = 0;

  public:
    /// location of the parsed token
    location loc;

    /// \name Ctor & dtor
    /// \{

    /**
     * \brief NmodlLexer constructor
     *
     * @param driver NmodlDriver where this lexer resides
     * @param in Input stream from where tokens will be read
     * @param out Output stream where output will be sent
     */
    explicit NmodlLexer(NmodlDriver& driver,
                        std::istream* in = nullptr,
                        std::ostream* out = nullptr)
        : NmodlFlexLexer(in, out)
        , driver(driver) {}

    ~NmodlLexer() override = default;

    /// \}

    /**
     * \brief Reset the column position of lexer to 0
     *
     * Due to COPY mode the end position is not accurate. Set column to 0 to
     * avoid confusion (see JIRA issue NOCMODL-25)
     */
    void reset_end_position() {
        loc.end.column = 0;
    }

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
    virtual NmodlParser::symbol_type next_token();

    /**
     * \brief Scan subsequent text as unit
     *
     * For units we have to consume string until end of closing parenthesis
     * and store it in the scanner. This will be later returned by get_unit().
     */
    void scan_unit();

    /**
     * \brief Input text until end of line
     *
     * For construct like TITLE we have to scan text until end of line
     */
    std::string input_line();

    /// Return last scanned unit as ast::String
    ast::String* get_unit();

    /// Enable debug output (via yyout) if compiled into the scanner.
    void set_debug(bool b);
};

/** @} */  // end of lexer

}  // namespace parser
}  // namespace nmodl

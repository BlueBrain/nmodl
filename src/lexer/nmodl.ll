%{
    #include <iostream>
    #include "ast/ast.hpp"
    #include "lexer/nmodl_lexer.hpp"
    #include "lexer/nmodl_utils.hpp"
    #include "lexer/token_mapping.hpp"
    #include "parser/nmodl_driver.hpp"
    #include "utils/string_utils.hpp"

    /** YY_USER_ACTION is called before each of token actions and
      * we update columns by length of the token. Node that position
      * starts at column 1 and increasing by yyleng gives extra position
      * larger than exact columns span of the token. Hence in ModToken
      * class we reduce the length by one column. */
    #define YY_USER_ACTION { loc.step(); loc.columns(yyleng); }

   /** By default yylex returns int, we use token_type. Unfortunately
     * yyterminate by default returns 0, which is not of token_type. */
   #define yyterminate() return nmodl::Parser::make_END(loc);

   /** Disables inclusion of unistd.h, which is not available under Visual
     * C++ on Win32. The C++ scanner uses STL streams instead. */
   #define YY_NO_UNISTD_H

%}


/** regex for digits and exponent for float */
D   [0-9]
E   [Ee][-+]?{D}+

/** we do use yymore feature in copy modes */
%option yymore

/** name of the lexer header file */
%option header-file="nmodl_base_lexer.hpp"

/** name of the lexer implementation file */
%option outfile="nmodl_base_lexer.cpp"

/** change the name of the scanner class (to "NmodlFlexLexer") */
%option prefix="Nmodl"

/** enable C++ scanner which is also reentrant */
%option c++

/** no plan for include files for now */
%option noyywrap

/** need to unput characters back to buffer for custom routines */
%option unput

/** need to put in buffer for custom routines */
%option input

/** not an interactive lexer, takes a file instead */
%option batch

/** enable debug mode (disable for production) */
%option debug

/** instructs flex to generate an 8-bit scanner, i.e.,
  * one which can recognize 8-bit characters. */
%option 8bit

/** show warning messages */
%option warn

/* to insure there are no holes in scanner rules */
%option nodefault

/* keep line information */
%option yylineno

/* mode for verbatim or comment */
%x COPY_MODE

/* mode for DEFINE variable name (i.e. macro definition) */
%x MACRO_NAME_MODE

/* mode for DEFINE variable value (i.e. macro definition) */
%x MACRO_VALUE_MODE

/* mode for TITLE and single line comment */
%x LINE_MODE

/* enable use of start condition stacks */
%option stack

%%


[a-zA-Z][a-zA-Z0-9_]*'+ {
                            /** Prime could high order and if it is followed by 0 then
                              * it's name. For this we have to consume extra character
                              * and then have to decide the token type. If it's actually
                              * prime then we have to un-put consumed character. */
                            auto nextch =  yyinput();
                            std::string name(yytext);

                            if(nextch == '0') {
                                name += "0";
                                return name_symbol(name, loc);

                            } else {
                                unput(nextch);
                                return prime_symbol(name, loc);
                            }
                        }

WHILE   |
IF      |
ELSE                    {
                            /** Lower or upper case if,else,while keywords are allowded.
                              * To avoid extra keywords, make token lower case */
                             for (char *ch = yytext; *ch; ++ch)
                                *ch = tolower(*ch);

                            return token_symbol(yytext, loc);
                        }

"VERBATIM"              {
                            /** start of verbatim block */
                            BEGIN(COPY_MODE);
                        }

"COMMENT"               {
                            /** start of comment block */
                            BEGIN(COPY_MODE);
                        }

"DEFINE"                {
                            /** start of macro definition */
                            BEGIN(MACRO_NAME_MODE);
                            return token_symbol(yytext, loc);
                        }

"TITLE"                 {
                            /** start of nmodl title. We return rest of line as LINE_PART. */
                            BEGIN(LINE_MODE);
                            return token_symbol(yytext, loc);
                        }

<MACRO_NAME_MODE>[a-zA-Z][a-zA-Z0-9_]* {

                            /** macro name (typically string) */
                            BEGIN(MACRO_VALUE_MODE);
                            return name_symbol(yytext, loc, Token::INTEGER);
                        }

<MACRO_VALUE_MODE>{D}+  {
                            /** macro value (typically integer) */
                            BEGIN(INITIAL);
                            return integer_symbol(atoi(yytext), loc);
                        }

[a-zA-Z][a-zA-Z0-9_]*   {
                            /** First check if token is nmodl keyword or method  name used in
                              * solve statements. */
                            if (is_keyword(yytext) || is_method(yytext)) {

                                /** Token for certain keywords need name_ptr value. */
                                auto type = token_type(yytext);
                                ModToken tok(yytext, type, loc);
                                auto value = new ast::Name( new ast::String(yytext) );
                                value->setToken(tok);

                                switch (static_cast<int>(type)) {
                                    /** Tokens requiring name_ptr as value */
                                    case Token::METHOD:
                                        return nmodl::Parser::make_METHOD(value, loc);
                                    case Token::SUFFIX:
                                        return nmodl::Parser::make_SUFFIX(value, loc);
                                    case Token::VALENCE:
                                        return nmodl::Parser::make_VALENCE(value, loc);
                                    case Token::DEL:
                                        return nmodl::Parser::make_DEL(value, loc);
                                    case Token::DEL2:
                                        return nmodl::Parser::make_DEL2(value, loc);

                                    /** We have to store context for the reaction type */
                                    case Token::NONLINEAR:
                                    case Token::LINEAR:
                                    case Token::PARTIAL:
                                    case Token::KINETIC:
                                        lexcontext = type;
                                        break;
                                }

                                /** value is not used */
                                delete value;
                                return token_symbol(yytext, loc, type);
                            } else {

                                /** Check if name is already defined as macro. If so, return token
                                  * as integer with token as it's name. Otherwise return it as
                                  * regular name token. */
                                if (driver.is_defined_var(yytext)) {
                                    auto value = driver.get_defined_var_value(yytext);
                                    return integer_symbol(value, loc, yytext);
                                } else {
                                    return name_symbol(yytext, loc);
                                }
                            }
                        }

{D}+                    {
                            return integer_symbol(atoi(yytext), loc);
                        }

{D}+"."{D}*({E})? |
{D}*"."{D}+({E})? |
{D}+{E}                 {
                            return double_symbol(atof(yytext), loc);
                        }

\"[^\"]*\"              {
                            /* check comment about can't quote \" */
                            return string_symbol(yytext, loc);
                        }

">"                     {
                            return token_symbol(yytext, loc, Token::GT);
                        }

">="                    {
                            return token_symbol(yytext, loc, Token::GE);
                        }

"<"                     {
                            return token_symbol(yytext, loc, Token::LT);
                        }

"<="                    {
                            return token_symbol(yytext, loc, Token::LE);
                        }

"=="                    {
                            return token_symbol(yytext, loc, Token::EQ);
                        }

"!="                    {
                            return token_symbol(yytext, loc, Token::NE);
                        }

"!"                     {
                            return token_symbol(yytext, loc, Token::NOT);
                        }

"&&"                    {
                            return token_symbol(yytext, loc, Token::AND);
                        }

"||"                    {
                            return token_symbol(yytext, loc, Token::OR);
                        }

"<->"                   {
                            return token_symbol(yytext, loc, Token::REACT1);
                        }

"~+"                    {
                            return token_symbol(yytext, loc, Token::NONLIN1);
                        }

"~"                     {
                            /** return token depending on the reaction context */
                            if (lexcontext == Token::NONLINEAR) {
                                return token_symbol(yytext, loc, Token::NONLIN1);
                            }

                            if (lexcontext == Token::LINEAR) {
                                return token_symbol(yytext, loc, Token::LIN1);
                            }

                            if (lexcontext == Token::PARTIAL) {
                                return token_symbol(yytext, loc, Token::TILDE);
                            }

                            if (lexcontext == Token::KINETIC) {
                                return token_symbol(yytext, loc, Token::REACTION);
                            }

                            /* \todo : should be parser error instead of exception */
                            auto msg = "Lexer Error : Invalid context, no token matched for ~";
                            throw std::runtime_error(msg);
                        }

"{"                     {
                            return token_symbol(yytext, loc, Token::OPEN_BRACE);
                        }

"}"                     {
                            return token_symbol(yytext, loc, Token::CLOSE_BRACE);
                        }

"("                     {
                            return token_symbol(yytext, loc, Token::OPEN_PARENTHESIS);
                        }

")"                     {
                            return token_symbol(yytext, loc, Token::CLOSE_PARENTHESIS);
                        }

"["                     {
                            return token_symbol(yytext, loc, Token::OPEN_BRACKET);
                        }

"]"                     {
                            return token_symbol(yytext, loc, Token::CLOSE_BRACKET);
                        }

"@"                     {
                            return token_symbol(yytext, loc, Token::AT);
                        }

"+"                     {
                            return token_symbol(yytext, loc, Token::ADD);
                        }

"-"                     {
                            return token_symbol(yytext, loc, Token::MINUS);
                        }

"*"                     {
                            return token_symbol(yytext, loc, Token::MULTIPLY);
                        }

"/"                     {
                            return token_symbol(yytext, loc, Token::DIVIDE);
                        }

"="                     {
                            return token_symbol(yytext, loc, Token::EQUAL);
                        }

"^"                     {
                            return token_symbol(yytext, loc, Token::CARET);
                        }

","                     {
                            return token_symbol(yytext, loc, Token::COMMA);
                        }

<MACRO_NAME_MODE>[ \t]  |
<MACRO_VALUE_MODE>[ \t] |
[ \t]                   {
                            loc.step();
                        }

\r\n                    {
                            loc.end.column = 1;
                            loc.step();
                            loc.lines(1);
                        }

\r                      {
                            loc.end.column = 1;
                            loc.step();
                        }

\n.*                    {
                            /** First we read entire line and print to stdout. This is useful
                              * for using lexer program. */
                            std::string str(yytext);
                            stringutils::trim(str);

                            if (driver.is_verbose()) {
                                if(str.length()) {
                                    stringutils::trimnewline(str);
                                    std::cout << "LINE "<< yylineno << ": " << str << std::endl;
                                } else {
                                    std::cout << "LINE " << yylineno << ": " << std::endl;
                                }
                            }

                            /** Pass back entire string except newline charactear */
                            yyless(1);

                            loc.lines(1);
                            loc.step();
                        }

:.* |
\?.*                    {
                            /** Todo : add grammar support for single line comment.
                              * Here yytext already has entire part of string */
                        }

.                       {
                            return token_symbol(yytext, loc, Token::PERIOD);
                        }

<COPY_MODE>[ \t]        {
                            yymore();
                        }

<COPY_MODE>\n           {
                            yymore();
                            loc.lines (1);
                        }

<COPY_MODE>\r\n         {
                            yymore();
                            loc.lines (1);
                        }

<COPY_MODE>\r           {
                            yymore();
                            loc.lines (1);
                        }

<COPY_MODE>"ENDVERBATIM" {
                            /** For verbatim block we construct entire block again, resent end
                             * column position to 0 and return token. We do same for comment. */
                            auto str = "VERBATIM" + std::string(yytext);
                            BEGIN(INITIAL);
                            reset_end_position();
                            return nmodl::Parser::make_VERBATIM(str, loc);
                         }

<COPY_MODE>"ENDCOMMENT" {
                            auto str = "COMMENT" + std::string(yytext);
                            BEGIN(INITIAL);
                            reset_end_position();
                            return nmodl::Parser::make_COMMENT(str, loc);
                        }

<COPY_MODE><<EOF>>      {
                            std::cout << "\n ERROR: Unexpected end of file in COPY_MODE! \n";
                            return nmodl::Parser::make_END(loc);
                        }

<COPY_MODE>.            {
                            yymore();
                        }

<LINE_MODE>\n   |
<LINE_MODE>\r\n         {
                            /** For title return string without new line character */
                            loc.lines(1);
                            std::string str(yytext);
                            stringutils::trimnewline(str);
                            BEGIN(INITIAL);
                            return nmodl::Parser::make_LINE_PART(str, loc);
                        }

<LINE_MODE>.            {
                            yymore();
                        }

\"[^\"\n]*$             {
                            std::cout << "\n ERROR: Unterminated string (e.g. for printf) \n";
                        }

%%


/** Some of the utility functions can't be defined outside due to macros.
  * These are utility functions ported from original nocmodl implementation. */


/** This implementation of NmodlFlexLexer::yylex() is required to fill the
  * vtable of the class NmodlFlexLexer. We define the scanner's main yylex
  * function via YY_DECL to reside in the Lexer class instead. */
int NmodlFlexLexer::yylex() {
  throw std::runtime_error("next_token() should be used instead of yylex()");
}

/** yy_flex_debug is member of parent scanner class */
void nmodl::Lexer::set_debug(bool b) {
    yy_flex_debug = b;
}

/** Scan unit which is a string between opening and closing parenthesis.
  * We store this in lexer itself and then consumed later from the parser. */
void nmodl::Lexer::scan_unit() {
    /** We are interested in unit after first parenthesis.
     * So to get correct position update the location. */
    loc.step();
    std::string str;

    /** Unit is a string until close parenthis */
    while (1) {
        auto lastch =  yyinput();
        if(lastch == ')') {
            unput(')');
            break;
        }
        else if ( lastch == '\n' || lastch == 0) {
            std::cout << "ERROR: While parsing unit, closing parenthis not found";
            break;
        }
        str += lastch;
    }

    /** YY_USER_ACTION is not executed if are consuming input
      * using yyinput and hence increase location */
    loc.columns(str.size());

    ModToken tok(str, Token::UNITS, loc);
    last_unit = new ast::String(str);
    last_unit->setToken(tok);
}

/** return last scanned unit, it shouln't be null pointer */
ast::string_ptr nmodl::Lexer::get_unit() {
    if (last_unit == nullptr) {
        throw std::runtime_error("Trying to get unscanned empty unit");
    }
    auto result = last_unit;
    last_unit = nullptr;
    return result;
}

/**********************************************************************************
 *
 * @brief Bison grammar and parser implementation for Units of Neuron like MOD2C
 *
 * Parsing Unit definition file for use in NMODL.
 *
 * CREDIT: This is based on parsing nrnunits.lib like MOD2C
 * https://github.com/BlueBrain/mod2c/blob/master/share/nrnunits.lib
 *****************************************************************************/

%code requires
{
    #include <string>
    #include <sstream>

    #include "parser/unit_driver.hpp"
    #include "units/units.hpp"
}

/** use C++ parser interface of bison */
%skeleton "lalr1.cc"

/** require modern bison version */
%require  "3.0.2"

/** print verbose error messages instead of just message 'syntax error' */
%define parse.error verbose

/** enable tracing parser for debugging */
%define parse.trace

/** add extra arguments to yyparse() and yylex() methods */
%parse-param {class UnitLexer& scanner}
%parse-param {class UnitDriver& driver}
%lex-param   {nmodl::UnitScanner &scanner}
%lex-param   {nmodl::UnitDriver &driver}

/** use variant based implementation of semantic values */
%define api.value.type variant

/** assert correct cleanup of semantic value objects */
%define parse.assert

/** handle symbol to be handled as a whole (type, value, and possibly location) in scanner */
%define api.token.constructor

/** specify the namespace for the parser class */
%define api.namespace {nmodl::parser}

/** set the parser's class identifier */
%define parser_class_name {UnitParser}

/** keep track of the current position within the input */
%locations

%token               END    0     "end of file"
%token               INVALID_TOKEN

%token <std::string> BASE_UNIT
%token <std::string> INVALID_BASE_UNIT
%token <std::string> UNIT
%token <std::string> UNIT_POWER
%token <std::string> PREFIX
%token <std::string> DOUBLE
%token <std::string> FRACTION
%token <std::string> COMMENT
%token <std::string> NEWLINE
%token <std::string> UNIT_PROD
%token <std::string> DIVISION

%type <std::vector<std::string>*> units_nom
%type <std::vector<std::string>*> units_denom
%type <nmodl::units::unit*> nominator
%type <nmodl::units::unit*> item
%type <nmodl::units::prefix*> prefix
%type <std::shared_ptr<nmodl::units::UnitTable>> list

%{
    #include "lexer/unit_lexer.hpp"
    #include "parser/unit_driver.hpp"

    using nmodl::parser::UnitParser;
    using nmodl::parser::UnitLexer;
    using nmodl::parser::UnitDriver;

    /// yylex takes scanner as well as driver reference
    /// \todo: check if driver argument is required
    static UnitParser::symbol_type yylex(UnitLexer &scanner, UnitDriver &/*driver*/) {
        return scanner.next_token();
    }
%}


%start list_option

%%

list_option
    : END
    | list END

list
    : {
        $$ = driver.Table;
      }
    | list item  {
        $1->insert($2);
        $$ = $1;
      }
    | list prefix  {
        $1->insert_prefix($2);
        $$ = $1;
      }
    | list no_insert {
        $$ = $1;
      }
    ;

units_nom
    : {
        $$ = new std::vector<std::string>;
      }
    | UNIT units_nom {
        $2->push_back($1);
        $$ = $2;
      }
    | UNIT_POWER units_nom {
        $2->push_back($1);
        $$ = $2;
      }
    | UNIT_PROD units_nom {
        $$ = $2;
      }
    ;

units_denom
    : {
        $$ = new std::vector<std::string>;
      }
    | UNIT units_denom {
        $2->push_back($1);
        $$ = $2;
      }
    | UNIT_POWER units_denom {
        $2->push_back($1);
        $$ = $2;
      }
    | UNIT_PROD units_denom {
        $$ = $2;
      }
    ;

nominator
    : units_nom {
        nmodl::units::unit* newunit = new nmodl::units::unit();
        newunit->add_nominator_unit($1);
        $$ = newunit;
      }
    | DOUBLE units_nom {
        nmodl::units::unit* newunit = new nmodl::units::unit();
        newunit->add_nominator_unit($2);
        newunit->add_nominator_double($1);
        $$ = newunit;
      }
    | FRACTION units_nom {
        nmodl::units::unit* newunit = new nmodl::units::unit();
        newunit->add_nominator_unit($2);
        newunit->add_fraction($1);
        $$ = newunit;
      }
    ;

prefix
    : PREFIX DOUBLE NEWLINE {
        $$ = new nmodl::units::prefix($1,$2);
      }
    | PREFIX UNIT NEWLINE {
        $$ = new nmodl::units::prefix($1,$2);
      }

no_insert
    : COMMENT NEWLINE
    | NEWLINE
    | INVALID_TOKEN {
        error(scanner.loc, "item");
      }
item
    : UNIT BASE_UNIT NEWLINE {
        nmodl::units::unit *newunit = new nmodl::units::unit($1);
        newunit->add_base_unit($2);
        $$ = newunit;
      }
    | UNIT INVALID_BASE_UNIT NEWLINE {
            error(scanner.loc, "Base units should be named by characters a-j");
          }
    | UNIT nominator NEWLINE {
        $2->add_unit($1);
        std::vector<std::string> nominator = $2->get_nominator_unit();
        $$ = $2;
      }
    | UNIT nominator DIVISION units_denom NEWLINE {
        $2->add_unit($1);
        std::vector<std::string> nominator = $2->get_nominator_unit();
        $2->add_denominator_unit($4);
        std::vector<std::string> denominator = $2->get_denominator_unit();
        $$ = $2;
      }
    ;

%%

/** Bison expects error handler for parser */

void UnitParser::error(const location &loc , const std::string &message) {
    std::stringstream ss;
    ss << "Unit Parser Error : " << message << " [Location : " << loc << "]";
    throw std::runtime_error(ss.str());
}
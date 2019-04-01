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
%token <std::string> UNIT
%token <std::string> UNIT_POWER
%token <std::string> PREFIX
%token <std::string> DOUBLE
%token <std::string> FRACTION
%token <std::string> COMMENT
%token <std::string> NEWLINE
%token <std::string> UNIT_PROD
%token <std::string> DIVISION

%type <unit::table*> units_nom
%type <unit::table*> units_denom
%type <unit::table*> nominator
%type <unit::table*> item
%type <unit::UnitTable*> list

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
    : END { std::cout << "end" << std::endl; }
    | list END { std::cout << "reset" << std::endl; if($1){ driver.Table.reset($1); } }

list
    : item {
        if($1){
            std::cout << "new UnitTable" << std::endl;
            $$ = new unit::UnitTable($1);
        }
        else{
            std::cout << "new empty UnitTable" << std::endl;
            $$ = new unit::UnitTable();
        }
     }
    | list item {
        if($2){
            std::cout << "insert item " << $2->get_name() << " " << $2->get_factor() << std::endl;
            $1->insert($2); $$ = $1;
         }
        else {
            std::cout << "insert nothing" << std::endl;
            $$ = $1;
        }
     }
    ;

units_nom
    : {
        std::cout << "new unit" << std::endl;
        unit::table *newunit = new unit::table();
        $$ = newunit;
      }
    | UNIT units_nom {
        std::cout << $1 << std::endl;
        $2->addUnit($1);
        $$ = $2;
      }
    | UNIT_POWER units_nom {
        std::cout << $1 << std::endl;
        $2->addUnit($1);
        $$ = $2;
      }
    | UNIT_PROD units_nom { $$ = $2; }
    ;

units_denom
    : { std::cout << "denominator" << std::endl; }
    | UNIT units_denom { std::cout << $1 << std::endl; }
    | UNIT_POWER units_denom { std::cout << $1 << std::endl; }
    | UNIT_PROD units_denom {}
    ;

nominator
    : units_nom { $$ = $1; }
    | DOUBLE units_nom {
        $2->addNominatorDouble($1);
        std::cout << "DOUBLE = " << $2->get_factor() << std::endl;
        $$ = $2;
      }
    | FRACTION units_nom {
        std::cout << "FRACTION = " << $1 << std::endl;
        $$ = $2;
      }
    ;

item
    : PREFIX DOUBLE NEWLINE {
        std::cout << "PREFIX " << $1 << std::endl;
        std::cout << "DOUBLE " << $2 << std::endl;
        $$ = nullptr;
      }
    | PREFIX UNIT NEWLINE {
        std::cout << "PREFIX " << $1 << std::endl;
        std::cout << "UNIT " << $2 << std::endl;
        $$ = nullptr;
      }
    | UNIT BASE_UNIT NEWLINE {
        std::cout << "UNIT " << $1 << std::endl;
        std::cout << "BASE UNIT " << $2 << std::endl;
        $$ = nullptr;
      }
    | UNIT nominator NEWLINE {
        $2->addUnit($1);
        std::cout << "newunit: " << $2->get_name() << ", " << $2->get_factor() << std::endl;
        $$ =$2;
      }
    | UNIT nominator DIVISION units_denom NEWLINE {
        std::cout << "UNIT " <<  "DIV" << std::endl;
        $$ = nullptr;
      }
    | COMMENT NEWLINE {
        std::cout << "COMMENT " << $1 << std::endl;
        $$ = nullptr;
      }
    | NEWLINE { $$ = nullptr; }
    | INVALID_TOKEN {
        error(scanner.loc, "item");
        $$ = nullptr;
      }
    ;

%%

/** Bison expects error handler for parser */

void UnitParser::error(const location &loc , const std::string &message) {
    std::stringstream ss;
    ss << "Unit Parser Error : " << message << " [Location : " << loc << "]";
    throw std::runtime_error(ss.str());
}
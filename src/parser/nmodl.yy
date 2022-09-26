/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 * Copyright (C) 2018-2022 Michael Hines
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

/**********************************************************************************
 *
 * @brief Bison grammar and parser implementation for NMODL
 *
 * This implementation is based NEURON's nmodl program. The grammar rules,
 * symbols, terminals and non-terminals closely resember to NEURON
 * implementation. This is to support entire NMODL language for NEURON as well
 * as CoreNEURON. As opposed to non-reentrant C parser, this implementation
 * uses C++ interface of flex and bison. This makes lexer/parser fully
 * reentrant.
 * One of the implementation decision was whether to construct/return smart
 * pointers from parser or let ast class constructor to handle creation of
 * smart pointers from raw one. After trying shared_ptr and unique_ptr
 * implementations within bison parset, the later approach seems more
 * convenient and flexible while working with large number of grammar rules.
 * But this could be change once ast specification and compiler passes will
 * be implemented.
 *****************************************************************************/

/** to include parser's header file, one has to include ast definitions */
 %code requires
 {
    #include "ast/all.hpp"
 }

/** use C++ parser interface of bison */
%skeleton "lalr1.cc"

/** require modern bison version */
%require  "3.0.2"

/** print verbose error messages instead of just message 'syntax error' */
%define parse.error verbose

/** enable tracing parser for debugging */
%define parse.trace

/** add extra arguments to yyparse() and yylexe() methods */
%parse-param {class NmodlLexer& scanner}
%parse-param {class NmodlDriver& driver}
%lex-param { nmodl::NmodlScanner &scanner }
%lex-param { nmodl::NmodlDriver &driver }

/** use variant based implementation of semantic values */
%define api.value.type variant

/** assert correct cleanup of semantic value objects */
%define parse.assert

/** handle symbol to be handled as a whole (type, value, and possibly location) in scanner */
%define api.token.constructor

/** specify the namespace for the parser class */
%define api.namespace {nmodl::parser}

/** set the parser's class identifier */
%define parser_class_name {NmodlParser}

/** keep track of the current position within the input */
%locations

/** Initializations before parsing : Use filename from driver to initialize location object */
%initial-action
{
    @$.begin.filename = @$.end.filename = &driver.stream_name;
};

/** Tokens for lexer : we return ModToken object from lexer. This is useful when
 *  we want to access original string and location. With C++ interface and other
 *  location related information, this is now less useful in parser. But when we
 *  lexer executable or tests, it's useful to return ModToken. Note that UNKNOWN
 *  token is added for convenience (with default arguments). */

%token  <ModToken>              AFTER
%token  <ModToken>              BBCOREPOINTER
%token  <ModToken>              BEFORE
%token  <ModToken>              BREAKPOINT
%token  <ModToken>              BY
%token  <ModToken>              COMPARTMENT
%token  <ModToken>              CONDUCTANCE
%token  <ModToken>              CONSERVE
%token  <ModToken>              CONSTANT
%token  <ModToken>              CONSTRUCTOR
%token  <ModToken>              DEFINE1
%token  <ModToken>              DEPEND
%token  <ModToken>              ASSIGNED
%token  <ModToken>              DERFUNC
%token  <ModToken>              DERIVATIVE
%token  <ModToken>              DESTRUCTOR
%token  <ModToken>              DISCRETE
%token  <ModToken>              ELECTRODE_CURRENT
%token  <ModToken>              ELSE
%token  <ModToken>              EQUATION
%token  <ModToken>              EXTERNAL
%token  <ModToken>              FIRST
%token  <ModToken>              FORALL1
%token  <ModToken>              FOR_NETCONS
%token  <ModToken>              FROM
%token  <ModToken>              FUNCTION1
%token  <ModToken>              FUNCTION_TABLE
%token  <ModToken>              GLOBAL
%token  <ModToken>              IF
%token  <ModToken>              IFERROR
%token  <ModToken>              INCLUDE1
%token  <ModToken>              INDEPENDENT
%token  <ModToken>              INITIAL1
%token  <ModToken>              KINETIC
%token  <ModToken>              LAG
%token  <ModToken>              LAST
%token  <ModToken>              LIN1
%token  <ModToken>              LINEAR
%token  <ModToken>              LOCAL
%token  <ModToken>              LONGDIFUS
%token  <ModToken>              MATCH
%token  <ModToken>              MODEL
%token  <ModToken>              MODEL_LEVEL
%token  <ModToken>              NETRECEIVE
%token  <ModToken>              NEURON
%token  <ModToken>              NONLIN1
%token  <ModToken>              NONLINEAR
%token  <ModToken>              NONSPECIFIC
%token  <ModToken>              NRNMUTEXLOCK
%token  <ModToken>              NRNMUTEXUNLOCK
%token  <ModToken>              PARAMETER
%token  <ModToken>              PARTIAL
%token  <ModToken>              PLOT
%token  <ModToken>              POINTER
%token  <ModToken>              PROCEDURE
%token  <ModToken>              PROTECT
%token  <ModToken>              RANGE
%token  <ModToken>              REACT1
%token  <ModToken>              REACTION
%token  <ModToken>              READ
%token  <ModToken>              RESET
%token  <ModToken>              SECTION
%token  <ModToken>              SENS
%token  <ModToken>              SOLVE
%token  <ModToken>              SOLVEFOR
%token  <ModToken>              START1
%token  <ModToken>              STATE
%token  <ModToken>              STEADYSTATE
%token  <ModToken>              STEP
%token  <ModToken>              STEPPED
%token  <ModToken>              SWEEP
%token  <ModToken>              TABLE
%token  <ModToken>              TERMINAL
%token  <ModToken>              THREADSAFE
%token  <ModToken>              TO
%token  <ModToken>              UNITS
%token  <ModToken>              UNITSOFF
%token  <ModToken>              UNITSON
%token  <ModToken>              USEION
%token  <ModToken>              USING
%token  <ModToken>              VS
%token  <ModToken>              WATCH
%token  <ModToken>              WHILE
%token  <ModToken>              WITH
%token  <ModToken>              WRITE

%token  <ModToken>              OR
%token  <ModToken>              AND
%token  <ModToken>              GT
%token  <ModToken>              LT
%token  <ModToken>              LE
%token  <ModToken>              EQ
%token  <ModToken>              NE
%token  <ModToken>              NOT
%token  <ModToken>              GE

%token  <ast::Double>           REAL
%token  <ast::Integer>          INTEGER
%token  <ast::Integer>          DEFINEDVAR

%token  <ast::Name>             NAME
%token  <ast::Name>             METHOD
%token  <ast::Name>             SUFFIX
%token  <ast::Name>             VALENCE
%token  <ast::Name>             DEL
%token  <ast::Name>             DEL2
%token  <ast::Name>             FLUX_VAR

%token  <ast::PrimeName>        PRIME

%token  <std::string>           VERBATIM
%token  <std::string>           BLOCK_COMMENT
%token  <std::string>           LINE_COMMENT
%token  <std::string>           LINE_PART

%token  <ast::String>           STRING

%token  <ModToken>              OPEN_BRACE          "{"
%token  <ModToken>              CLOSE_BRACE         "}"
%token  <ModToken>              OPEN_PARENTHESIS    "("
%token  <ModToken>              CLOSE_PARENTHESIS   ")"
%token  <ModToken>              OPEN_BRACKET        "["
%token  <ModToken>              CLOSE_BRACKET       "]"
%token  <ModToken>              AT                  "@"
%token  <ModToken>              ADD                 "+"
%token  <ModToken>              MULTIPLY            "*"
%token  <ModToken>              MINUS               "-"
%token  <ModToken>              DIVIDE              "/"
%token  <ModToken>              EQUAL               "="
%token  <ModToken>              CARET               "^"
%token  <ModToken>              COLON               ":"
%token  <ModToken>              COMMA               ","
%token  <ModToken>              TILDE               "~"
%token  <ModToken>              PERIOD              "."

%token  <ModToken>              REPRESENTS
%token  <std::string>           ONTOLOGY_ID

%token  END                     0                   "End of file"
%token                          UNKNOWN
%token                          INVALID_TOKEN


/** Bison production return types */


%type   <ast::Program*>                     all

%type   <ast::Double*>                      double
%type   <ast::Identifier*>                  name
%type   <ast::Integer*>                     integer
%type   <ast::Name*>                        Name
%type   <ast::Number*>                      NUMBER
%type   <ast::Number*>                      number
%type   <ast::VarName*>                     variable_name

%type   <ast::Model*>                       model
%type   <ast::Unit*>                        units
%type   <ast::Integer*>                     optional_index
%type   <ast::Unit*>                        unit
%type   <ast::Block*>                       procedure
%type   <ast::Limits*>                      limits
%type   <ast::Double*>                      abs_tolerance
%type   <ast::Expression*>                  integer_expression
%type   <ast::Expression*>                  primary
%type   <ast::Expression*>                  term
%type   <ast::Expression*>                  left_linear_expression
%type   <ast::Expression*>                  linear_expression
%type   <ast::NumberVector>                 number_list
%type   <ast::Expression*>                  expression
%type   <ast::Expression*>                  watch_expression
%type   <ast::Statement*>                   statement_type1
%type   <ast::Statement*>                   statement_type2
%type   <ast::StatementBlock*>              statement_list
%type   <ast::LocalListStatement*>          local_statement
%type   <ast::LocalVarVector>               local_var_list
%type   <ast::ExpressionVector>             expression_list
%type   <ast::Define*>                      define
%type   <ast::Expression*>                  assignment
%type   <ast::FromStatement*>               from_statement
%type   <ast::WhileStatement*>              while_statement
%type   <ast::IfStatement*>                 if_statement
%type   <ast::ElseIfStatementVector>        optional_else_if
%type   <ast::ElseStatement*>               optional_else
%type   <ast::WrappedExpression*>           function_call
%type   <ast::StatementBlock*>              if_solution_error
%type   <ast::Expression*>                  optional_increment
%type   <ast::Number*>                      optional_start
%type   <ast::VarNameVector>                sens_list
%type   <ast::Sens*>                        sens
%type   <ast::LagStatement*>                lag_statement
%type   <ast::ForAllStatement*>             forall_statement
%type   <ast::ParamAssign*>                 parameter_assignment
%type   <ast::Stepped*>                     stepped_statement
%type   <ast::IndependentDefinition*>       independent_definition
%type   <ast::AssignedDefinition*>          dependent_definition
%type   <ast::Block*>                       declare
%type   <ast::ParamAssignVector>            parameter_block_body
%type   <ast::IndependentDefinitionVector>  independent_block_body
%type   <ast::AssignedDefinitionVector>     dependent_block_body
%type   <ast::SteppedVector>                step_block_body
%type   <ast::WatchStatement*>              watch_statement
%type   <ast::BinaryOperator>               watch_direction
%type   <ast::Watch*>                       watch
%type   <ast::ForNetcon*>                   for_netcon
%type   <ast::PlotDeclaration*>             plot_declaration
%type   <ast::PlotVarVector>                plot_variable_list
%type   <ast::ConstantStatementVector>      constant_statement
%type   <ast::MatchVector>                  match_list
%type   <ast::Match*>                       match
%type   <ast::Identifier*>                  match_name
%type   <ast::PartialBoundary*>             partial_equation
%type   <ast::FirstLastTypeIndex*>          first_last
%type   <ast::ReactionStatement*>           reaction_statement
%type   <ast::Conserve*>                    conserve
%type   <ast::Expression*>                  react
%type   <ast::Compartment*>                 compartment
%type   <ast::LonDifuse*>                   longitudinal_diffusion
%type   <ast::NameVector>                   name_list
%type   <ast::ExpressionVector>             unit_block_body
%type   <ast::UnitDef*>                     unit_definition
%type   <ast::FactorDef*>                   factor_definition
%type   <ast::NameVector>                   optional_solvefor
%type   <ast::NameVector>                   solvefor
%type   <ast::UnitState*>                   unit_state
%type   <ast::NameVector>                   optional_table_var_list
%type   <ast::NameVector>                   table_var_list
%type   <ast::TableStatement*>              table_statement
%type   <ast::NameVector>                   optional_dependent_var_list
%type   <ast::ArgumentVector>               optional_argument_list
%type   <ast::ArgumentVector>               argument_list
%type   <ast::Integer*>                     optional_array_index
%type   <ast::Useion*>                      use_ion_statement
%type   <ast::StatementVector>              neuron_statement
%type   <ast::ReadIonVarVector>             read_ion_list
%type   <ast::WriteIonVarVector>            write_ion_list
%type   <ast::String*>                      ontology
%type   <ast::NonspecificCurVarVector>      nonspecific_var_list
%type   <ast::ElectrodeCurVarVector>        electrode_current_var_list
%type   <ast::SectionVarVector>             section_var_list
%type   <ast::RangeVarVector>               range_var_list
%type   <ast::GlobalVarVector>              global_var_list
%type   <ast::PointerVarVector>             pointer_var_list
%type   <ast::BbcorePointerVarVector>       bbcore_pointer_var_list
%type   <ast::ExternVarVector>              external_var_list
%type   <ast::ThreadsafeVarVector>          optional_threadsafe_var_list
%type   <ast::ThreadsafeVarVector>          threadsafe_var_list
%type   <ast::Valence*>                     valence
%type   <ast::ExpressionStatement*>         initial_statement
%type   <ast::ConductanceHint*>             conductance
%type   <ast::StatementVector>              optional_statement_list
%type   <ast::WrappedExpression*>           flux_variable

%type   <ast::AssignedBlock*>               dependent_block
%type   <ast::BABlock*>                     before_after_block
%type   <ast::BreakpointBlock*>             breakpoint_block
%type   <ast::ConstantBlock*>               constant_block
%type   <ast::ConstructorBlock*>            constructor_block
%type   <ast::DerivativeBlock*>             derivative_block
%type   <ast::DestructorBlock*>             destructor_block
%type   <ast::DiscreteBlock*>               discrete_block
%type   <ast::FunctionBlock*>               function_block
%type   <ast::FunctionTableBlock*>          function_table_block
%type   <ast::IndependentBlock*>            independent_block
%type   <ast::InitialBlock*>                initial_block
%type   <ast::KineticBlock*>                kinetic_block
%type   <ast::LinearBlock*>                 linear_block
%type   <ast::MatchBlock*>                  match_block
%type   <ast::NetReceiveBlock*>             net_receive_block
%type   <ast::NeuronBlock*>                 neuron_block
%type   <ast::NonLinearBlock*>              non_linear_block
%type   <ast::ParamBlock*>                  parameter_block
%type   <ast::PartialBlock*>                partial_block
%type   <ast::ProcedureBlock*>              procedure_block
%type   <ast::SolveBlock*>                  solve_block
%type   <ast::StateBlock*>                  state_block
%type   <ast::StepBlock*>                   step_block
%type   <ast::TerminalBlock*>               terminal_block
%type   <ast::UnitBlock*>                   unit_block

%type   <ast::Integer*>                     INTEGER_PTR
%type   <ast::Name*>                        NAME_PTR
%type   <ast::String*>                      STRING_PTR

/*
 * Precedence and Associativity : specify operator precedency and
 * associativity (from lower to higher. Note that '^' represent
 * exponentiation.
 */

%left   OR
%left   AND
%left   GT GE LT LE EQ NE
%left   "+" "-"
%left   "*" "/" "%"
%left   UNARYMINUS NOT
%right  "^"

%{
    #include "lexer/nmodl_lexer.hpp"
    #include "parser/nmodl_driver.hpp"
    #include "parser/verbatim_driver.hpp"

    using nmodl::parser::NmodlParser;
    using nmodl::parser::NmodlLexer;
    using nmodl::parser::NmodlDriver;
    using nmodl::parser::VerbatimDriver;

    /// yylex takes scanner as well as driver reference
    /// \todo Check if driver argument is required
    static NmodlParser::symbol_type yylex(NmodlLexer &scanner, NmodlDriver &/*driver*/) {
        return scanner.next_token();
    }

    /// forward declaration for the function which handles interface with other parsers
    std::string parse_with_verbatim_parser(std::string);

    // void nmodl parser tracing function
    void nmodl_trace_rule(const nmodl::parser::location &loc, size_t id, const std::string& rule);
%}

/** start symbol */
%start top


%%

/** Grammar rules : specification of all grammar rules for NMODL. The core
 *  grammar rules are based on NEURON implementation but all other
 *  implementation details are changed to support generate AST. Some rules
 *  are adapted to support better error handling and location tracking.
 *  Note that YYLLOC_DEFAULT is not sufficient for all rules and hence
 *  we need to update @$ (especially @$.begin). Consider below example:
 *
 *  neuron_statement RANGE range_var_list
 *
 *  In this case we have to do : @$.begin = $1.begin (@$.end is already
 *  set to $3.end and hence don't need to update).
 *
 *  \todo ModToken is set for the symbols returned by Lexer. But we need to
 *  set accurate location for each production. We need to add method in AST
 *  classes to handle this.
 *
 *  \todo LINE_COMMENT adds comment as separate statement and hence they
 *   go into separate line in nmodl printer. Need to update grammar to distinguish
 *   standalone single line comment vs. inline comment.
 */

top             :   all
                    {
                        nmodl_trace_rule(scanner.loc, 1, "all");
                        driver.set_ast($1);
                    }
                |   error
                    {
                        error(scanner.loc, "top");
                    }
                ;


all             :   {
                        nmodl_trace_rule(scanner.loc, 2, "");
                        $$ = new ast::Program();
                    }
                |   all model
                    {
                        nmodl_trace_rule(scanner.loc, 3, "all model");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all local_statement
                    {
                        nmodl_trace_rule(scanner.loc, 4, "all local_statement");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all define
                    {
                        nmodl_trace_rule(scanner.loc, 5, "all define");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all declare
                    {
                        nmodl_trace_rule(scanner.loc, 6, "all declare");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all MODEL_LEVEL INTEGER_PTR declare
                    {
                        /** todo This is discussed with Michael Hines. Model level was inserted
                         * by merge program which is no longer exist. This was to avoid the name
                         * collision in case of include. Idea was to have some kind of namespace!
                         */
                        nmodl_trace_rule(scanner.loc, 7, "all MODEL_LEVEL INTEGER_PTR declare");
                    }
                |   all procedure
                    {
                        nmodl_trace_rule(scanner.loc, 8, "all procedure");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all VERBATIM
                    {
                        nmodl_trace_rule(scanner.loc, 9, "all VERBATIM");
                        auto text = parse_with_verbatim_parser($2);
                        auto statement = new ast::Verbatim(new ast::String(text));
                        $1->emplace_back_node(statement);
                        $$ = $1;
                    }
                |   all BLOCK_COMMENT
                    {
                        nmodl_trace_rule(scanner.loc, 10, "all BLOCK_COMMENT");
                        auto text = parse_with_verbatim_parser($2);
                        auto statement = new ast::BlockComment(new ast::String(text));
                        $1->emplace_back_node(statement);
                        $$ = $1;
                    }
                |   all LINE_COMMENT
                    {
                        nmodl_trace_rule(scanner.loc, 11, "all LINE_COMMENT");
                        auto statement = new ast::LineComment(new ast::String($2));
                        $1->emplace_back_node(statement);
                        $$ = $1;
                    }
                |   all unit_state
                    {
                        nmodl_trace_rule(scanner.loc, 12, "all unit_state");
                        $1->emplace_back_node($2);
                        $$ = $1;
                    }
                |   all INCLUDE1 STRING_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 13, "all INCLUDE1 STRING_PTR");
                        $1->emplace_back_node(driver.parse_include(driver.check_include_argument(scanner.loc, $3->get_value()), scanner.loc));
                        $$ = $1;
                    }
                ;


model           :   MODEL LINE_PART
                    {
                        nmodl_trace_rule(scanner.loc, 14, "MODEL LINE_PART");
                        $$ = new ast::Model(new ast::String($2));
                    }
                ;


define         :    DEFINE1 NAME_PTR INTEGER_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 15, "DEFINE1 NAME_PTR INTEGER_PTR");
                        $$ = new ast::Define($2, $3);
                        driver.add_defined_var($2->get_node_name(), $3->eval());
                    }
                |   DEFINE1 error
                    {
                        error(scanner.loc, "define");
                    }
                ;


Name            :   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 16, "NAME_PTR");
                        $$ = $1;
                    }
                ;


declare         :   parameter_block
                    {
                        nmodl_trace_rule(scanner.loc, 17, "parameter_block");
                        $$ = $1;
                    }
                |   independent_block
                    {
                        nmodl_trace_rule(scanner.loc, 18, "independent_block");
                        $$ = $1;
                    }
                |   dependent_block
                    {
                        nmodl_trace_rule(scanner.loc, 19, "dependent_block");
                        $$ = $1;
                    }
                |   state_block
                    {
                        nmodl_trace_rule(scanner.loc, 20, "state_block");
                        $$ = $1;
                    }
                |   step_block
                    {
                        nmodl_trace_rule(scanner.loc, 21, "step_block");
                        $$ = $1;
                    }
                |   plot_declaration
                    {
                        nmodl_trace_rule(scanner.loc, 22, "plot_declaration");
                        $$ = new ast::PlotBlock($1);
                    }
                |   neuron_block
                    {
                        nmodl_trace_rule(scanner.loc, 23, "neuron_block");
                        $$ = $1;
                    }
                |   unit_block
                    {
                        nmodl_trace_rule(scanner.loc, 24, "unit_block");
                        $$ = $1;
                    }
                |   constant_block
                    {
                        nmodl_trace_rule(scanner.loc, 25, "constant_block");
                        $$ = $1;
                    }
                ;


parameter_block :   PARAMETER "{" parameter_block_body"}"
                    {
                        nmodl_trace_rule(scanner.loc, 26, "PARAMETER \"{\" parameter_block_body\"}\"");
                        $$ = new ast::ParamBlock($3);
                    }
                ;


parameter_block_body :
                    {
                        nmodl_trace_rule(scanner.loc, 27, "");
                        $$ = ast::ParamAssignVector();
                    }
                |   parameter_block_body parameter_assignment
                    {
                        nmodl_trace_rule(scanner.loc, 28, "parameter_block_body parameter_assignment");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


parameter_assignment :  NAME_PTR "=" number units limits
                    {
                        nmodl_trace_rule(scanner.loc, 29, "NAME_PTR \"=\" number units limits");
                        $$ = new ast::ParamAssign($1, $3, $4, $5);
                    }
                |   NAME_PTR units limits
                    {
                        nmodl_trace_rule(scanner.loc, 30, "NAME_PTR units limits");
                        $$ = new ast::ParamAssign($1, NULL, $2, $3);
                    }
                |   NAME_PTR "[" integer "]" units limits
                    {
                        nmodl_trace_rule(scanner.loc, 31, "NAME_PTR \"[\" integer \"]\" units limits");
                        $$ = new ast::ParamAssign(new ast::IndexedName($1, $3), NULL, $5, $6);
                    }
                |   error
                    {
                        error(scanner.loc, "parameter_assignment");
                    }
                ;


units           :   {
                        nmodl_trace_rule(scanner.loc, 32, "");
                        $$ = nullptr;
                    }
                |   unit
                    {
                        nmodl_trace_rule(scanner.loc, 33, "unit");
                        $$ = $1;
                    }
                ;


unit            :   "(" { scanner.scan_unit(); } ")"
                    {
                        nmodl_trace_rule(scanner.loc, 34, "\"(\" { scanner.scan_unit(); } \")\"");
                        // @todo Empty units should be handled in semantic analysis
                        auto unit = scanner.get_unit();
                        auto text = unit->eval();
                        $$ = new ast::Unit(unit);
                    }
                ;


unit_state      :   UNITSON
                    {
                        nmodl_trace_rule(scanner.loc, 35, "UNITSON");
                        $$ = new ast::UnitState(ast::UNIT_ON);
                    }
                |   UNITSOFF
                    {
                        nmodl_trace_rule(scanner.loc, 36, "UNITSOFF");
                        $$ = new ast::UnitState(ast::UNIT_OFF);
                    }
                ;


limits          :   {
                        nmodl_trace_rule(scanner.loc, 37, "");
                        $$ = nullptr;
                    }
                |   LT double "," double GT
                    {
                        nmodl_trace_rule(scanner.loc, 38, "LT double \",\" double GT");
                        $$ = new ast::Limits($2, $4);
                    }
                ;


step_block      :   STEPPED "{" step_block_body "}"
                    {
                        nmodl_trace_rule(scanner.loc, 39, "STEPPED \"{\" step_block_body \"}\"");
                        $$ = new ast::StepBlock($3);
                    }
                ;


step_block_body :   {
                        nmodl_trace_rule(scanner.loc, 40, "");
                        $$ = ast::SteppedVector();
                    }
                |   step_block_body stepped_statement
                    {
                        nmodl_trace_rule(scanner.loc, 41, "step_block_body stepped_statement");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


stepped_statement : NAME_PTR "=" number_list units
                    {
                        nmodl_trace_rule(scanner.loc, 42, "NAME_PTR \"=\" number_list units");
                        $$ = new ast::Stepped($1, $3, $4);
                    }
                ;


number_list     :   number "," number
                    {
                        nmodl_trace_rule(scanner.loc, 43, "number \",\" number");
                        $$ = ast::NumberVector();
                        $$.emplace_back($1);
                        $$.emplace_back($3);
                    }
                |   number_list "," number
                    {
                        nmodl_trace_rule(scanner.loc, 44, "number_list \",\" number");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                ;


name            :   Name
                    {
                        nmodl_trace_rule(scanner.loc, 45, "Name");
                        $$ = $1;
                    }
                |   PRIME
                    {
                        nmodl_trace_rule(scanner.loc, 46, "PRIME");
                        $$ = $1.clone();
                    }
                ;


number          :   NUMBER
                    {
                        nmodl_trace_rule(scanner.loc, 47, "NUMBER");
                        $$ = $1;
                    }
                |   "-" NUMBER
                    {
                        nmodl_trace_rule(scanner.loc, 48, "\"-\" NUMBER");
                        $2->negate();
                        $$ = $2;
                    }
                |   "+" NUMBER
                    {
                        nmodl_trace_rule(scanner.loc, 49, "\"+\" NUMBER");
                        $$ = $2;
                    }
                ;


NUMBER          :   integer
                    {
                        nmodl_trace_rule(scanner.loc, 50, "integer");
                        $$ = $1;
                    }
                |   REAL
                    {
                        nmodl_trace_rule(scanner.loc, 51, "REAL");
                        $$ = $1.clone();
                    }
                ;


integer         :   INTEGER_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 52, "INTEGER_PTR");
                        $$ = $1;
                    }
                |   DEFINEDVAR
                    {
                        nmodl_trace_rule(scanner.loc, 53, "DEFINEDVAR");
                        $$ = $1.clone();
                    }
                ;


double          :   REAL
                    {
                        nmodl_trace_rule(scanner.loc, 54, "REAL");
                        $$ = $1.clone();
                    }
                |   integer
                    {
                        nmodl_trace_rule(scanner.loc, 55, "integer");
                        $$ = new ast::Double(std::to_string(($1->eval())));
                        delete($1);
                    }
                ;


independent_block : INDEPENDENT "{" independent_block_body "}"
                    {
                        nmodl_trace_rule(scanner.loc, 56, "INDEPENDENT \"{\" independent_block_body \"}\"");
                        $$ = new ast::IndependentBlock($3);
                        ModToken block_token = $1 + $4;
                        $$->set_token(block_token);
                    }
                ;


independent_block_body :
                    {
                        nmodl_trace_rule(scanner.loc, 57, "");
                        $$ = ast::IndependentDefinitionVector();
                    }
                |   independent_block_body independent_definition
                    {
                        nmodl_trace_rule(scanner.loc, 58, "independent_block_body independent_definition");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                |   independent_block_body SWEEP independent_definition
                    {
                        nmodl_trace_rule(scanner.loc, 59, "independent_block_body SWEEP independent_definition");
                        $1.emplace_back($3);
                        $3->set_sweep(std::make_shared<ast::Boolean>(1));
                        $$ = $1;
                    }
                ;


independent_definition :  NAME_PTR FROM number TO number withby integer optional_start units
                    {
                        nmodl_trace_rule(scanner.loc, 60, "NAME_PTR FROM number TO number withby integer optional_start units");
                        $$ = new ast::IndependentDefinition(NULL, $1, $3, $5, $7, $8, $9);
                    }
                |   error
                    {
                        error(scanner.loc, "independent_definition");
                    }
                ;


withby          :   WITH
                ;


dependent_block :   ASSIGNED "{" dependent_block_body "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::AssignedBlock($3);
                    }
                ;


dependent_block_body :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::AssignedDefinitionVector();
                    }
                |   dependent_block_body dependent_definition
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


dependent_definition : name optional_start units abs_tolerance
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::AssignedDefinition($1, NULL, NULL, NULL, $2, $3, $4);
                    }
                |   name "[" integer "]" optional_start units abs_tolerance
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::AssignedDefinition($1, $3, NULL, NULL, $5, $6, $7);
                    }
                |   name FROM number TO number optional_start units abs_tolerance
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::AssignedDefinition($1, NULL, $3, $5, $6, $7, $8);
                    }
                |   name "[" integer "]" FROM number TO number optional_start units abs_tolerance
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::AssignedDefinition($1, $3, $6, $8, $9, $10, $11);
                    }
                |   error
                    {
                        error(scanner.loc, "dependent_definition");
                    }
                ;


optional_start  :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   START1 number
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


abs_tolerance   :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   LT double GT
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


state_block     :   STATE  "{" dependent_block_body "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::StateBlock($3);
                    }
                ;


plot_declaration :  PLOT plot_variable_list VS name optional_index
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::PlotDeclaration($2, new ast::PlotVar($4,$5));
                    }
                |   PLOT error
                    {
                        error(scanner.loc, "plot_declaration");
                    }
                ;


plot_variable_list : name optional_index
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::PlotVarVector();
                        auto variable = new ast::PlotVar($1, $2);
                        $$.emplace_back(variable);
                    }
                |   plot_variable_list "," name optional_index
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                        auto variable = new ast::PlotVar($3, $4);
                        $$.emplace_back(variable);
                    }
                ;


optional_index  :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   "[" INTEGER_PTR "]"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


procedure       :   initial_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   derivative_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   breakpoint_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   linear_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   non_linear_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   function_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   procedure_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   net_receive_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   terminal_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   discrete_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   partial_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   kinetic_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   constructor_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   destructor_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   function_table_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   BEFORE before_after_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto new_before_block = new ast::BeforeBlock($2);
                        ModToken block_token = $1+*($2->get_token());
                        new_before_block->set_token(block_token);
                        $$ = new_before_block;
                    }
                |   AFTER before_after_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto new_after_block = new ast::AfterBlock($2);
                        ModToken block_token = $1+*($2->get_token());
                        new_after_block->set_token(block_token);
                        $$ = new_after_block;
                    }
                ;


initial_block   :   INITIAL1 statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::InitialBlock($2);
                    }
                ;


constructor_block : CONSTRUCTOR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ConstructorBlock($2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                ;


destructor_block :  DESTRUCTOR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::DestructorBlock($2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                ;


statement_list  :   "{" optional_statement_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::StatementBlock($2);
                        $$->set_token($1);
                    }
                |   "{" local_statement optional_statement_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $3.insert($3.begin(), std::shared_ptr<ast::LocalListStatement>($2));
                        $$ = new ast::StatementBlock($3);
                        $$->set_token($1);
                    }
                ;


conductance     :   CONDUCTANCE Name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ConductanceHint($2, NULL);
                    }
                |   CONDUCTANCE Name USEION NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ConductanceHint($2, $4);
                    }
                ;


local_statement :   LOCAL local_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LocalListStatement($2);
                    }
                |   LOCAL error
                    {
                        error(scanner.loc, "local_statement");
                    }
                ;


local_var_list  :   NAME_PTR optional_array_index
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::LocalVarVector();
                        if($2) {
                            auto variable = new ast::LocalVar(new ast::IndexedName($1, $2));
                            $$.emplace_back(variable);
                        } else {
                            auto variable = new ast::LocalVar($1);
                            $$.emplace_back(variable);
                        }
                    }
                |   local_var_list "," NAME_PTR optional_array_index
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        if($4) {
                            auto variable = new ast::LocalVar(new ast::IndexedName($3, $4));
                            $1.emplace_back(variable);
                        } else {
                            auto variable = new ast::LocalVar($3);
                            $1.emplace_back(variable);
                        }
                        $$ = $1;
                    }
                ;


optional_array_index :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   "[" integer "]"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


optional_statement_list :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::StatementVector();
                    }
                |   optional_statement_list statement_type1
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                |   optional_statement_list statement_type2
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                |   optional_statement_list LINE_COMMENT
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto statement = new ast::LineComment(new ast::String($2));
                        $1.emplace_back(statement);
                        $$ = $1;
                    }
                ;


statement_type1 :   from_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   forall_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   while_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   if_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                |   solve_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                |   conductance
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   VERBATIM
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto text = parse_with_verbatim_parser($1);
                        $$ = new ast::Verbatim(new ast::String(text));
                    }
                |   BLOCK_COMMENT
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto text = parse_with_verbatim_parser($1);
                        $$ = new ast::BlockComment(new ast::String(text));
                    }
                |   sens
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   compartment
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   longitudinal_diffusion
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   conserve
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   lag_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   RESET
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Reset();
                    }
                |   match_block
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                |   partial_equation
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   table_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   unit_state
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   initial_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   watch_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   for_netcon
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                |   NRNMUTEXLOCK
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::MutexLock();
                    }
                |   NRNMUTEXUNLOCK
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::MutexUnlock();
                    }
                |   error
                    {
                        error(scanner.loc, "statement_type1");
                    }
                ;


statement_type2 :   assignment
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                |   PROTECT assignment
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ProtectStatement($2);
                    }
                |   reaction_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   function_call
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement($1);
                    }
                ;


assignment      :   variable_name "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_ASSIGN), $3);
                        auto name = $1->get_name();
                        if ((name->is_prime_name()) ||
                            (name->is_indexed_name() &&
                            std::dynamic_pointer_cast<ast::IndexedName>(name)->get_name()->is_prime_name()))
                        {
                            $$ = new ast::DiffEqExpression(expression);
                        } else {
                            $$ = expression;
                        }
                    }
                |   nonlineqn expression "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::NonLinEquation($2, $4);
                    }
                |   lineqn left_linear_expression "=" linear_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LinEquation($2, $4);
                    }
                ;


variable_name   :   name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::VarName($1, nullptr, nullptr);
                    }
                |   name "[" integer_expression "]"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::VarName(new ast::IndexedName($1, $3), nullptr, nullptr);
                    }
                |   NAME_PTR "@" integer
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::VarName($1, $3, nullptr);
                    }
                |   NAME_PTR "@" integer "[" integer_expression "]"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::VarName($1, $3, $5);
                    }
                ;


integer_expression : Name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   integer
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   "(" integer_expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::ParenExpression($2);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   integer_expression "+" integer_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_ADDITION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   integer_expression "-" integer_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_SUBTRACTION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   integer_expression "*" integer_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_MULTIPLICATION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   integer_expression "/" integer_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_DIVISION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   error
                    {
                        error(scanner.loc, "integer_expression");
                    }
                ;


expression      :   variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   flux_variable
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   double units
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        if($2)
                            $$ = new ast::DoubleUnit($1, $2);
                        else
                            $$ = $1;
                    }
                |   function_call
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   "(" expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::ParenExpression($2);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression "+" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression  = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_ADDITION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression "-" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression  = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_SUBTRACTION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression "*" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression  = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_MULTIPLICATION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression "/" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression  = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_DIVISION), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression "^" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression  = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_POWER), $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                |   expression OR expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_OR), $3);
                    }
                |   expression AND expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_AND), $3);
                    }
                |   expression GT expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_GREATER), $3);
                    }
                |   expression LT expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_LESS), $3);
                    }
                |   expression GE expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_GREATER_EQUAL), $3);
                    }
                |   expression LE expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_LESS_EQUAL), $3);
                    }
                |   expression EQ expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_EXACT_EQUAL), $3);
                    }
                |   expression NE expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_NOT_EQUAL), $3);
                    }
                |   NOT expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnaryExpression(ast::UnaryOperator(ast::UOP_NOT), $2);
                    }
                |   "-" expression %prec UNARYMINUS
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnaryExpression(ast::UnaryOperator(ast::UOP_NEGATION), $2);
                    }
                |   error
                    {
                        error(scanner.loc, "expression");
                    }
                ;

                /**
                    \todo Add extra rules for better error reporting

                | "(" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        yyerror("Unbalanced left parenthesis followed by valid expressions");
                    }
                | "(" error
                    {
                        yyerror("Unbalanced left parenthesis followed by non parseable");
                    }
                |  expression ")"
                    {
                        yyerror("Unbalanced right parenthesis");
                    }
                */


nonlineqn       : NONLIN1
                ;


lineqn          : LIN1
                ;


left_linear_expression : linear_expression
                {
                    nmodl_trace_rule(scanner.loc, 2, "error");
                    $$ = $1;
                }
                ;


linear_expression : primary
                {
                    nmodl_trace_rule(scanner.loc, 2, "error");
                    $$ = $1;
                }
                |   "-" primary
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnaryExpression(ast::UnaryOperator(ast::UOP_NEGATION), $2);
                    }
                |   linear_expression "+" primary
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_ADDITION), $3);
                    }
                |   linear_expression "-" primary
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_SUBTRACTION), $3);
                    }
                ;


primary         :   term
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   primary "*" term
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_MULTIPLICATION), $3);
                    }
                |   primary "/" term
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_DIVISION), $3);
                    }
                ;


term            :   variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   double
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   function_call
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   "(" expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ParenExpression($2);
                    }
                |   error
                    {
                        error(scanner.loc, "term");
                    }
                ;


function_call   :   NAME_PTR "(" expression_list ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto expression = new ast::FunctionCall($1, $3);
                        $$ = new ast::WrappedExpression(expression);
                    }
                ;


expression_list :   {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ExpressionVector();
                    }
                |   expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ExpressionVector();
                        $$.emplace_back($1);
                    }
                |   STRING_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ExpressionVector();
                        $$.emplace_back($1);
                    }
                |   expression_list "," expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                |   expression_list "," STRING_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                ;


from_statement  :   FROM NAME_PTR "=" integer_expression TO integer_expression optional_increment statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FromStatement($2, $4, $6, $7, $8);
                    }
                |   FROM error
                    {
                        error(scanner.loc, "from_statement");
                    }
                ;


optional_increment :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                | BY integer_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


forall_statement :  FORALL1 NAME_PTR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ForAllStatement($2, $3);
                    }
                |   FORALL1 error
                    {
                        error(scanner.loc, "forall_statement");
                    }
                ;


while_statement :   WHILE "(" expression ")" statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::WhileStatement($3, $5);
                    }
                ;


if_statement    :   IF "(" expression ")" statement_list "}" optional_else_if optional_else
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::IfStatement($3, $5, $7, $8);
                    }
                ;


optional_else_if :  {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ElseIfStatementVector();
                    }
                |   optional_else_if ELSE IF "(" expression ")" statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ElseIfStatement($5, $7));
                        $$ = $1;
                    }
                ;


optional_else   :   {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   ELSE statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ElseStatement($2);
                    }
                ;


derivative_block :  DERIVATIVE NAME_PTR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::DerivativeBlock($2, $3);
                        $$->set_token($1);
                    }
                ;


linear_block    :   LINEAR NAME_PTR optional_solvefor statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LinearBlock($2, $3, $4);
                        $$->set_token($1);
                    }
                ;


non_linear_block :  NONLINEAR NAME_PTR optional_solvefor statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::NonLinearBlock($2, $3, $4);
                        $$->set_token($1);
                    }
                ;


discrete_block  :   DISCRETE NAME_PTR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::DiscreteBlock($2, $3);
                        ModToken block_token = $1 + $4;
                        $$->set_token(block_token);
                    }
                ;


partial_block   :   PARTIAL NAME_PTR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::PartialBlock($2, $3);
                        ModToken block_token = $1 + $4;
                        $$->set_token(block_token);
                    }
                |   PARTIAL error
                    {
                        error(scanner.loc, "partial_block");
                    }
                ;


partial_equation :  "~" PRIME "=" NAME_PTR "*" DEL2 "(" NAME_PTR ")" "+" NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::PartialBoundary(NULL, $2.clone(), NULL, NULL, $4, $6.clone(), $8, $11);
                    }
                |   "~" DEL NAME_PTR "[" first_last "]" "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::PartialBoundary($2.clone(), $3, $5, $8, NULL, NULL, NULL, NULL);
                    }
                |   "~" NAME_PTR "[" first_last "]" "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::PartialBoundary(NULL, $2, $4, $7, NULL, NULL, NULL, NULL);
                    }
                ;


first_last      :   FIRST
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FirstLastTypeIndex(ast::PEQ_FIRST);
                    }
                |   LAST
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FirstLastTypeIndex(ast::PEQ_LAST);
                    }
                ;


function_table_block : FUNCTION_TABLE NAME_PTR "(" optional_argument_list ")" units
                {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FunctionTableBlock($2, $4, $6);
                        // units don't have token, use ")" as end location
                        ModToken block_token = $1 + $5;
                        $$->set_token(block_token);
                }
                ;


function_block  :   FUNCTION1 NAME_PTR "(" optional_argument_list ")" units statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FunctionBlock($2, $4, $6, $7);
                        $$->set_token($1);
                    }
                ;


optional_argument_list :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ArgumentVector();
                    }
                |   argument_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                ;


argument_list   :   name units
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ArgumentVector();
                        $$.emplace_back(new ast::Argument($1, $2));
                    }
                |   argument_list "," name units
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Argument($3, $4));
                        $$ = $1;
                    }
                ;


procedure_block :   PROCEDURE NAME_PTR "(" optional_argument_list ")" units statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ProcedureBlock($2, $4, $6, $7); $$->set_token($1);
                    }
                ;


net_receive_block : NETRECEIVE "(" optional_argument_list ")" statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::NetReceiveBlock($3, $5);
                    }
                |   NETRECEIVE error
                    {
                        error(scanner.loc, "net_receive_block");
                    }
                ;


initial_statement : INITIAL1 statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ExpressionStatement(new ast::InitialBlock($2));
                    }
                ;


solve_block     :   SOLVE NAME_PTR if_solution_error
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::SolveBlock($2, NULL, NULL, $3);
                        $$->set_token(*($2->get_token()));
                    }
                |   SOLVE NAME_PTR USING METHOD if_solution_error
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::SolveBlock($2, $4.clone(), NULL, $5);
                        $$->set_token(*($2->get_token()));
                    }
                |
                    SOLVE NAME_PTR STEADYSTATE METHOD if_solution_error
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::SolveBlock($2, NULL, $4.clone(), $5);
                        $$->set_token(*($2->get_token()));
                    }
                |   SOLVE error
                    {
                        error(scanner.loc, "solve_block");
                    }
                ;


if_solution_error :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   IFERROR statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


optional_solvefor :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                    }
                |   solvefor
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                ;


solvefor        :   SOLVEFOR NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                        $$.emplace_back($2);
                    }
                |   solvefor "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                |   SOLVEFOR error
                    {
                        error(scanner.loc, "solvefor");
                    }
                ;


breakpoint_block :  BREAKPOINT statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BreakpointBlock($2);
                    }
                ;


terminal_block  :   TERMINAL statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::TerminalBlock($2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                ;


before_after_block : BREAKPOINT statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BABlock(new ast::BABlockType(ast::BATYPE_BREAKPOINT), $2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                |   SOLVE statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BABlock(new ast::BABlockType(ast::BATYPE_SOLVE), $2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                |   INITIAL1 statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BABlock(new ast::BABlockType(ast::BATYPE_INITIAL), $2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                |   STEP statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BABlock(new ast::BABlockType(ast::BATYPE_STEP), $2);
                        ModToken block_token = $1 + $3;
                        $$->set_token(block_token);
                    }
                |   error
                    {
                        error(scanner.loc, "before_after_block");
                    }
                ;


watch_statement :   WATCH watch
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::WatchStatement(ast::WatchVector());
                        $$->emplace_back_watch($2);
                    }
                |   watch_statement "," watch
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1->emplace_back_watch($3); $$ = $1;
                    }
                |   WATCH error
                    {
                        error(scanner.loc, "watch_statement");
                    }
                ;


watch           :   "(" watch_expression watch_direction watch_expression ")" double
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Watch( new ast::BinaryExpression($2, $3, $4), $6);
                    }
                ;


watch_direction :   GT
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::BinaryOperator(ast::BOP_GREATER);
                    }
                |   LT
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::BinaryOperator(ast::BOP_LESS);
                    }
                ;


for_netcon      :   FOR_NETCONS "(" optional_argument_list ")" statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ForNetcon($3, $5);
                    }
                |   FOR_NETCONS error
                    {
                        error(scanner.loc, "for_netcon");
                    }
                ;


watch_expression :  variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   double units
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::DoubleUnit($1, $2);
                    }
                |   function_call
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   "(" watch_expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ParenExpression($2);
                    }
                |   watch_expression "+" watch_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_ADDITION), $3);
                    }
                |   watch_expression "-" watch_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_SUBTRACTION), $3);
                    }
                |   watch_expression "*" watch_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_MULTIPLICATION), $3);
                    }
                |   watch_expression "/" watch_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_DIVISION), $3);
                    }
                |   watch_expression "^" watch_expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::BinaryExpression($1, ast::BinaryOperator(ast::BOP_POWER), $3);
                    }
                |   "-" watch_expression %prec UNARYMINUS
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnaryExpression(ast::UnaryOperator(ast::UOP_NEGATION), $2);
                    }
                |   error
                    {
                        error(scanner.loc, "watch_expression");
                    }
                ;


sens            :   SENS sens_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Sens($2);
                    }
                |   SENS error
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        error(scanner.loc, "sens");
                    }
                ;


sens_list       :   variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::VarNameVector();
                        $$.emplace_back($1);
                    }
                |   sens_list "," variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                ;


conserve        :   CONSERVE react "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Conserve($2, $4);
                    }
                |   CONSERVE error
                    {
                        error(scanner.loc, "conserve");
                    }
                ;


compartment     :   COMPARTMENT NAME_PTR "," expression "{" name_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Compartment($2, $4, $6);
                    }
                |   COMPARTMENT expression "{" name_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Compartment(NULL, $2, $4);
                    }
                ;


longitudinal_diffusion : LONGDIFUS NAME_PTR "," expression "{" name_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LonDifuse($2, $4, $6);
                    }
                |   LONGDIFUS expression "{" name_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LonDifuse(NULL, $2, $4);
                    }
                ;


name_list       :   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                        $$.emplace_back($1);
                    }
                |   name_list NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


kinetic_block   :   KINETIC NAME_PTR optional_solvefor statement_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::KineticBlock($2, $3, $4);
                        $$->set_token($1);
                    }
                ;


reaction_statement : REACTION react REACT1 react "(" expression "," expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::ReactionOperator(ast::LTMINUSGT);
                        $$ = new ast::ReactionStatement($2, op, $4, $6, $8);
                    }
                |   REACTION react LT LT  "(" expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::ReactionOperator(ast::LTLT);
                        $$ = new ast::ReactionStatement($2, op, NULL, $6, NULL);
                    }
                |   REACTION react "-" GT "(" expression ")"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::ReactionOperator(ast::MINUSGT);
                        $$ = new ast::ReactionStatement($2, op, NULL, $6, NULL);
                    }
                |   REACTION error
                    {
                        /** \todo Need to revisit reaction_statement implementation */
                    }
                ;


react           :   variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ReactVarName(nullptr, $1);
                    }
                |   integer variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ReactVarName($1, $2);
                    }
                |   react "+" variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::BinaryOperator(ast::BOP_ADDITION);
                        auto variable = new ast::ReactVarName(nullptr, $3);
                        $$ = new ast::BinaryExpression($1, op, variable);
                    }
                |   react "+" integer variable_name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::BinaryOperator(ast::BOP_ADDITION);
                        auto variable = new ast::ReactVarName($3, $4);
                        $$ = new ast::BinaryExpression($1, op, variable);
                    }
                ;


lag_statement   :   LAG name BY NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::LagStatement($2, $4);
                    }
                |   LAG error
                    {
                        error(scanner.loc, "lag_statement");
                    }
                ;


match_block     :   MATCH "{" match_list "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::MatchBlock($3);
                        ModToken block_token = $1 + $4;
                        $$->set_token(block_token);
                    }
                ;


match_list      :   match
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::MatchVector();
                        $$.emplace_back($1);
                    }
                |   match_list match
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


match           :   name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Match($1, NULL);
                    }
                |   match_name "(" expression ")" "=" expression
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto op = ast::BinaryOperator(ast::BOP_ASSIGN);
                        auto lhs = new ast::ParenExpression($3);
                        auto rhs = $6;
                        auto expression = new ast::BinaryExpression(lhs, op, rhs);
                        $$ = new ast::Match($1, expression);
                    }
                |   error
                    {
                        error(scanner.loc, "match ");
                    }
                ;


match_name      :   name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                |   name "[" NAME_PTR "]"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::IndexedName($1, $3);
                    }
                ;


unit_block      :   UNITS "{" unit_block_body "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnitBlock($3);
                    }
                ;


unit_block_body :   {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ExpressionVector();
                    }
                |   unit_block_body unit_definition
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                |   unit_block_body factor_definition
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                ;


unit_definition :   unit "=" unit
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::UnitDef($1, $3);
                    }
                |   unit error
                    {
                        error(scanner.loc, "unit_definition ");
                    }
                ;


factor_definition : NAME_PTR "=" double unit
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FactorDef($1, $3, $4, NULL, NULL);
                        $$->set_token(*($1->get_token()));
                    }
                |   NAME_PTR "=" unit unit
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FactorDef($1, NULL, $3, NULL, $4);
                        $$->set_token(*($1->get_token()));
                    }
                |   NAME_PTR "=" unit "-" GT unit
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::FactorDef($1, NULL, $3, new ast::Boolean(1), $6);
                        $$->set_token(*($1->get_token()));
                    }
                |   error
                    {

                    }
                ;


constant_block  :   CONSTANT "{" constant_statement "}"
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::ConstantBlock($3);
                        ModToken block_token = $1 + $4;
                        $$->set_token(block_token);
                    }
                ;


constant_statement :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ConstantStatementVector();
                    }
                |   constant_statement NAME_PTR "=" number units
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto constant = new ast::ConstantVar($2, $4, $5);
                        $1.emplace_back(new ast::ConstantStatement(constant));
                        $$ = $1;
                    }
                ;


table_statement :   TABLE optional_table_var_list optional_dependent_var_list FROM expression TO expression WITH INTEGER_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::TableStatement($2, $3, $5, $7, $9);
                    }
                |   TABLE error
                    {
                        error(scanner.loc, "table_statement");
                    }
                ;


optional_table_var_list :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                    }
                |   table_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                ;


table_var_list  :   Name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                        $$.emplace_back($1);
                    }
                |   table_var_list "," Name
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($3);
                        $$ = $1;
                    }
                ;


optional_dependent_var_list :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NameVector();
                    }
                |   DEPEND table_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $2;
                    }
                ;


neuron_block    :   NEURON OPEN_BRACE neuron_statement CLOSE_BRACE
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto block = new ast::StatementBlock($3);
                        ModToken statement_block = $2 + $4;
                        block->set_token(statement_block);
                        $$ = new ast::NeuronBlock(block);
                        ModToken neuron_block = $1 + statement_block;
                        $$->set_token(neuron_block);
                    }
                ;


neuron_statement :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::StatementVector();
                    }
                |   neuron_statement SUFFIX NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Suffix($2.clone(), $3));
                        $$ = $1;
                    }
                |   neuron_statement use_ion_statement
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back($2);
                        $$ = $1;
                    }
                |   neuron_statement NONSPECIFIC nonspecific_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Nonspecific($3));
                        $$ = $1;
                    }
                |   neuron_statement ELECTRODE_CURRENT electrode_current_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ElectrodeCurrent($3));
                        $$ = $1;
                    }
                |   neuron_statement SECTION section_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Section($3));
                        $$ = $1;
                    }
                |   neuron_statement RANGE range_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Range($3));
                        $$ = $1;
                    }
                |   neuron_statement GLOBAL global_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Global($3));
                        $$ = $1;
                    }
                |   neuron_statement POINTER pointer_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::Pointer($3));
                        $$ = $1;
                    }
                |   neuron_statement BBCOREPOINTER bbcore_pointer_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::BbcorePointer($3));
                        $$ = $1;
                    }
                |   neuron_statement EXTERNAL external_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::External($3));
                        $$ = $1;
                    }
                |   neuron_statement THREADSAFE optional_threadsafe_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ThreadSafe($3));
                        $$ = $1;
                    }
                |   neuron_statement REPRESENTS ONTOLOGY_ID
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::OntologyStatement(new ast::String($3)));
                        $$ = $1;
                    }
                ;


use_ion_statement : USEION NAME_PTR READ read_ion_list valence ontology
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Useion($2, $4, ast::WriteIonVarVector(), $5, $6);
                    }
                |   USEION NAME_PTR WRITE write_ion_list valence ontology
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Useion($2, ast::ReadIonVarVector(), $4, $5, $6);
                    }
                |   USEION NAME_PTR READ read_ion_list WRITE write_ion_list valence ontology
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Useion($2, $4, $6, $7, $8);
                    }
                |   USEION error
                    {
                        error(scanner.loc, "use_ion_statement");
                    }
                ;


read_ion_list   :   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ReadIonVarVector();
                        $$.emplace_back(new ast::ReadIonVar($1));
                    }
                |   read_ion_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ReadIonVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "read_ion_list");
                    }
                ;


write_ion_list  :   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::WriteIonVarVector();
                        $$.emplace_back(new ast::WriteIonVar($1));
                    }
                |   write_ion_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::WriteIonVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "write_ion_list");
                    }
                ;


valence         :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   VALENCE double
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::Valence($1.clone(), $2);
                    }
                |   VALENCE "-" double
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $3->negate();
                        $$ = new ast::Valence($1.clone(), $3);
                    }
                ;


ontology        :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = nullptr;
                    }
                |   REPRESENTS ONTOLOGY_ID
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::String($2);
                    }
                ;


nonspecific_var_list : NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::NonspecificCurVarVector();
                        $$.emplace_back(new ast::NonspecificCurVar($1));
                    }
                |   nonspecific_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::NonspecificCurVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "nonspecific_var_list");
                    }
                ;


electrode_current_var_list : NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ElectrodeCurVarVector();
                        $$.emplace_back(new ast::ElectrodeCurVar($1));
                    }
                |   electrode_current_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ElectrodeCurVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "electrode_current_var_list");
                    }
                ;


section_var_list :  NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::SectionVarVector();
                        $$.emplace_back(new ast::SectionVar($1));
                    }
                |   section_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::SectionVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "section_var_list");
                    }
                ;


range_var_list  :   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::RangeVarVector();
                        $$.emplace_back(new ast::RangeVar($1));
                    }
                |   range_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::RangeVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "range_var_list");
                    }
                ;


global_var_list:   NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::GlobalVarVector();
                        auto new_global_var = new ast::GlobalVar($1);
                        new_global_var->set_token(*($1->get_token()));
                        $$.emplace_back(new_global_var);
                    }
                |   global_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto new_global_var = new ast::GlobalVar($3);
                        new_global_var->set_token(*($3->get_token()));
                        $1.emplace_back(new_global_var);
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "global_var_list");
                    }
                ;


pointer_var_list :  NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::PointerVarVector();
                        auto new_pointer_var = new ast::PointerVar($1);
                        new_pointer_var->set_token(*($1->get_token()));
                        $$.emplace_back(new_pointer_var);
                    }
                |   pointer_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        auto new_pointer_var = new ast::PointerVar($3);
                        new_pointer_var->set_token(*($3->get_token()));
                        $1.emplace_back(new_pointer_var);
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "pointer_var_list");
                    }
                ;


bbcore_pointer_var_list : NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::BbcorePointerVarVector();
                        $$.emplace_back(new ast::BbcorePointerVar($1));
                    }
                |   bbcore_pointer_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::BbcorePointerVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "bbcore_pointer_var_list");
                    }
                ;


external_var_list : NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ExternVarVector();
                        $$.emplace_back(new ast::ExternVar($1));
                    }
                |   external_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ExternVar($3));
                        $$ = $1;
                    }
                |   error
                    {
                        error(scanner.loc, "external_var_list");
                    }
                ;


optional_threadsafe_var_list :
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ThreadsafeVarVector();
                    }
                |   threadsafe_var_list
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1;
                    }
                ;


threadsafe_var_list : NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = ast::ThreadsafeVarVector();
                        $$.emplace_back(new ast::ThreadsafeVar($1));
                    }
                |   threadsafe_var_list "," NAME_PTR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $1.emplace_back(new ast::ThreadsafeVar($3));
                        $$ = $1;
                    }
                ;

 INTEGER_PTR    :   INTEGER
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1.clone();
                    }
                 ;


 NAME_PTR        :  NAME
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1.clone();
                    }
                 ;


 STRING_PTR      :  STRING
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = $1.clone();
                    }
                 ;

 flux_variable   :  FLUX_VAR
                    {
                        nmodl_trace_rule(scanner.loc, 2, "error");
                        $$ = new ast::WrappedExpression($1.clone());
                    }
                 ;
%%


/** Parse verbatim and commnet blocks : In the future we will use C parser to
 *  analyze verbatim blocks for better code generation. For GPU code generation
 *  we need to disable printf like statements. We could add new pragma
 *  annotations to enable certain optimizations. Idea of having separate parser
 *  is to support this type of analysis in separate parser. Currently we have
 *  "empty" Verbatim parser which scan and return same string. */

std::string parse_with_verbatim_parser(std::string str) {
    std::istringstream is(str.c_str());

    VerbatimDriver extcontext(&is);
    Verbatim_parse(&extcontext);

    std::string ss(*(extcontext.result));

    return ss;
}

void nmodl_trace_rule(const nmodl::parser::location &loc, size_t id, const std::string& rule) {
    std::cout << "NMODL PARSER TRACE: ID " << id << ". RULE " << rule << " in FILE " << loc.begin.filename  << " @ " << loc << std::endl;
}

/** Bison expects error handler for parser.
 *  \todo Need to implement error codes and driver should accumulate
 *  and report all errors. For now simply abort.
 */

void NmodlParser::error(const location &loc , const std::string &msg) {
    driver.parse_error(scanner, loc, msg);
}

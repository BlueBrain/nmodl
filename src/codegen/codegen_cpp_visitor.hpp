/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \dir
 * \brief Code generation backend implementations for CoreNEURON
 *
 * \file
 * \brief \copybrief nmodl::codegen::CodegenCppVisitor
 */

#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "codegen/codegen_info.hpp"
#include "codegen/codegen_naming.hpp"
#include "printer/code_printer.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

/// encapsulates code generation backend implementations
namespace nmodl {

namespace codegen {

/**
 * \defgroup codegen Code Generation Implementation
 * \brief Implementations of code generation backends
 *
 * \defgroup codegen_details Codegen Helpers
 * \ingroup codegen
 * \brief Helper routines/types for code generation
 * \{
 */

/**
 * \enum BlockType
 * \brief Helper to represent various block types
 *
 * Note: do not assign integers to these enums
 *
 */
enum class BlockType {
    /// initial block
    Initial,

    /// constructor block
    Constructor,

    /// destructor block
    Destructor,

    /// breakpoint block
    Equation,

    /// derivative block
    State,

    /// watch block
    Watch,

    /// net_receive block
    NetReceive,

    /// before / after block
    BeforeAfter,

    /// fake ending block type for loops on the enums. Keep it at the end
    BlockTypeEnd
};


/**
 * \enum MemberType
 * \brief Helper to represent various variables types
 *
 */
enum class MemberType {
    /// index / int variables
    index,

    /// range / double variables
    range,

    /// global variables
    global,

    /// thread variables
    thread
};


/**
 * \class IndexVariableInfo
 * \brief Helper to represent information about index/int variables
 *
 */
struct IndexVariableInfo {
    /// symbol for the variable
    const std::shared_ptr<symtab::Symbol> symbol;

    /// if variable resides in vdata field of NrnThread
    /// typically true for bbcore pointer
    bool is_vdata = false;

    /// if this is pure index (e.g. style_ion) variables is directly
    /// index and shouldn't be printed with data/vdata
    bool is_index = false;

    /// if this is an integer (e.g. tqitem, point_process) variable which
    /// is printed as array accesses
    bool is_integer = false;

    /// if the variable is qualified as constant (this is property of IndexVariable)
    bool is_constant = false;

    explicit IndexVariableInfo(std::shared_ptr<symtab::Symbol> symbol,
                               bool is_vdata = false,
                               bool is_index = false,
                               bool is_integer = false)
        : symbol(std::move(symbol))
        , is_vdata(is_vdata)
        , is_index(is_index)
        , is_integer(is_integer) {}
};


/**
 * \class ShadowUseStatement
 * \brief Represents ion write statement during code generation
 *
 * Ion update statement needs use of shadow vectors for certain backends
 * as atomics operations are not supported on cpu backend.
 *
 * \todo If shadow_lhs is empty then we assume shadow statement not required
 */
struct ShadowUseStatement {
    std::string lhs;
    std::string op;
    std::string rhs;
};

/** \} */  // end of codegen_details


using printer::CodePrinter;


/**
 * \defgroup codegen_backends Codegen Backends
 * \ingroup codegen
 * \brief Code generation backends for CoreNEURON
 * \{
 */

/**
 * \class CodegenCppVisitor
 * \brief %Visitor for printing C++ code compatible with legacy api of CoreNEURON
 *
 * \todo
 *  - Handle define statement (i.e. macros)
 *  - If there is a return statement in the verbatim block
 *    of inlined function then it will be error. Need better
 *    error checking. For example, see netstim.mod where we
 *    have removed return from verbatim block.
 */
class CodegenCppVisitor: public visitor::ConstAstVisitor {
  protected:
    using SymbolType = std::shared_ptr<symtab::Symbol>;


    /**
     * Code printer object for target (C++)
     */
    std::unique_ptr<CodePrinter> printer;


    /**
     * Name of mod file (without .mod suffix)
     */
    std::string mod_filename;


    /**
     * Data type of floating point variables
     */
    std::string float_type = codegen::naming::DEFAULT_FLOAT_TYPE;


    /**
     * Flag to indicate if visitor should avoid ion variable copies
     */
    bool optimize_ionvar_copies = true;


    /**
     * All ast information for code generation
     */
    codegen::CodegenInfo info;


    /**
     * Symbol table for the program
     */
    symtab::SymbolTable* program_symtab = nullptr;


    /**
     * All float variables for the model
     */
    std::vector<SymbolType> codegen_float_variables;


    /**
     * All int variables for the model
     */
    std::vector<IndexVariableInfo> codegen_int_variables;


    /**
     * All global variables for the model
     * \todo: this has become different than CodegenInfo
     */
    std::vector<SymbolType> codegen_global_variables;


    /**
     * Flag to indicate if visitor should print the visited nodes
     */
    bool codegen = false;


    /**
     * Variable name should be converted to instance name (but not for function arguments)
     */
    bool enable_variable_name_lookup = true;


    /**
     * \c true if currently net_receive block being printed
     */
    bool printing_net_receive = false;


    /**
     * \c true if currently initial block of net_receive being printed
     */
    bool printing_net_init = false;


    /**
     * \c true if currently printing top level verbatim blocks
     */
    bool printing_top_verbatim_blocks = false;


    /**
     * \c true if internal method call was encountered while processing verbatim block
     */
    bool internal_method_call_encountered = false;


    /**
     * Index of watch statement being printed
     */
    int current_watch_statement = 0;


    /**
     * Return Nmodl language version
     * \return A version
     */
    std::string nmodl_version() const noexcept {
        return codegen::naming::NMODL_VERSION;
    }


    /**
     * Name of the simulator the code was generated for
     */
    virtual std::string simulator_name() = 0;


    /**
     * Data type for the local variables
     */
    const char* local_var_type() const noexcept {
        return codegen::naming::DEFAULT_LOCAL_VAR_TYPE;
    }


    /**
     * Convert a given \c double value to its string representation
     * \param value The number to convert given as string as it is parsed by the modfile
     * \return      Its string representation
     */
    std::string format_double_string(const std::string& value);


    /**
     * Convert a given \c float value to its string representation
     * \param value The number to convert given as string as it is parsed by the modfile
     * \return      Its string representation
     */
    std::string format_float_string(const std::string& value);


    /**
     * Determine variable name in the structure of mechanism properties
     *
     * \param name         Variable name that is being printed
     * \param use_instance Should the variable be accessed via instance or data array
     * \return             The C++ string representing the access to the variable in the neuron
     * thread structure
     */
    virtual std::string get_variable_name(const std::string& name,
                                          bool use_instance = true) const = 0;


    /**
     * Check if function or procedure node has parameter with given name
     *
     * \tparam T Node type (either procedure or function)
     * \param node AST node (either procedure or function)
     * \param name Name of parameter
     * \return True if argument with name exist
     */
    template <typename T>
    bool has_parameter_of_name(const T& node, const std::string& name);


    /**
     * Rename function/procedure arguments that conflict with default arguments
     */
    void rename_function_arguments();


    /**
     * Arguments for "_threadargs_" macro in neuron implementation
     */
    virtual std::string nrn_thread_arguments() const = 0;


    /**
     * Process a verbatim block for possible variable renaming
     * \param text The verbatim code to be processed
     * \return     The code with all variables renamed as needed
     */
    virtual std::string process_verbatim_text(std::string const& text) = 0;


    /**
     * Print call to internal or external function
     * \param node The AST node representing a function call
     */
    virtual void print_function_call(const ast::FunctionCall& node) = 0;

    /**
     * Print atomic update pragma for reduction statements
     */
    virtual void print_atomic_reduction_pragma() = 0;


    /**
     * Check if given statement should be skipped during code generation
     * \param node The AST Statement node to check
     * \return     \c true if this Statement is to be skipped
     */
    static bool statement_to_skip(const ast::Statement& node);


    /**
     * Check if a semicolon is required at the end of given statement
     * \param node The AST Statement node to check
     * \return     \c true if this Statement requires a semicolon
     */
    static bool need_semicolon(const ast::Statement& node);


    /**
     * Print any statement block in nmodl with option to (not) print braces
     *
     * The individual statements (of type nmodl::ast::Statement) in the StatementBlock are printed
     * by accepting \c this visistor.
     *
     * \param node        A (possibly empty) statement block AST node
     * \param open_brace  Print an opening brace if \c false
     * \param close_brace Print a closing brace if \c true
     */
    void print_statement_block(const ast::StatementBlock& node,
                               bool open_brace = true,
                               bool close_brace = true);


    /**
     * Print the items in a vector as a list
     *
     * This function prints a given vector of elements as a list with given separator onto the
     * current printer. Elements are expected to be of type nmodl::ast::Ast and are printed by being
     * visited. Care is taken to omit the separator after the the last element.
     *
     * \tparam T The element type in the vector, which must be of type nmodl::ast::Ast
     * \param  elements The vector of elements to be printed
     * \param  separator The separator string to print between all elements
     * \param  prefix A prefix string to print before each element
     */
    template <typename T>
    void print_vector_elements(const std::vector<T>& elements,
                               const std::string& separator,
                               const std::string& prefix = "");


    /// This constructor is private, only the derived classes' public constructors are public
    CodegenCppVisitor(std::string mod_filename,
                      const std::string& output_dir,
                      std::string float_type,
                      const bool optimize_ionvar_copies)
        : printer(std::make_unique<CodePrinter>(output_dir + "/" + mod_filename + ".cpp"))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}


    /// This constructor is private, only the derived classes' public constructors are public
    CodegenCppVisitor(std::string mod_filename,
                      std::ostream& stream,
                      std::string float_type,
                      const bool optimize_ionvar_copies)
        : printer(std::make_unique<CodePrinter>(stream))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}


    void visit_binary_expression(const ast::BinaryExpression& node) override;
    void visit_binary_operator(const ast::BinaryOperator& node) override;
    void visit_boolean(const ast::Boolean& node) override;
    void visit_double(const ast::Double& node) override;
    void visit_else_if_statement(const ast::ElseIfStatement& node) override;
    void visit_else_statement(const ast::ElseStatement& node) override;
    void visit_float(const ast::Float& node) override;
    void visit_from_statement(const ast::FromStatement& node) override;
    void visit_function_call(const ast::FunctionCall& node) override;
    void visit_if_statement(const ast::IfStatement& node) override;
    void visit_indexed_name(const ast::IndexedName& node) override;
    void visit_integer(const ast::Integer& node) override;
    void visit_local_list_statement(const ast::LocalListStatement& node) override;
    void visit_name(const ast::Name& node) override;
    void visit_paren_expression(const ast::ParenExpression& node) override;
    void visit_prime_name(const ast::PrimeName& node) override;
    void visit_statement_block(const ast::StatementBlock& node) override;
    void visit_string(const ast::String& node) override;
    void visit_unary_operator(const ast::UnaryOperator& node) override;
    void visit_unit(const ast::Unit& node) override;
    void visit_var_name(const ast::VarName& node) override;
    void visit_verbatim(const ast::Verbatim& node) override;
    void visit_while_statement(const ast::WhileStatement& node) override;
    void visit_update_dt(const ast::UpdateDt& node) override;
    void visit_protect_statement(const ast::ProtectStatement& node) override;
    void visit_mutex_lock(const ast::MutexLock& node) override;
    void visit_mutex_unlock(const ast::MutexUnlock& node) override;
};


template <typename T>
void CodegenCppVisitor::print_vector_elements(const std::vector<T>& elements,
                                              const std::string& separator,
                                              const std::string& prefix) {
    for (auto iter = elements.begin(); iter != elements.end(); iter++) {
        printer->add_text(prefix);
        (*iter)->accept(*this);
        if (!separator.empty() && !nmodl::utils::is_last(iter, elements)) {
            printer->add_text(separator);
        }
    }
}

/** \} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl
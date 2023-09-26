/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \dir
 * \brief Code generation backend implementations for NEURON
 *
 * \file
 * \brief \copybrief nmodl::codegen::CodegenNeuronCppVisitor
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
#include <codegen/codegen_cpp_visitor.hpp>


/// encapsulates code generation backend implementations
namespace nmodl {

namespace codegen {


using printer::CodePrinter;


/**
 * \defgroup codegen_backends Codegen Backends
 * \ingroup codegen
 * \brief Code generation backends for NEURON
 * \{
 */

/**
 * \class CodegenNeuronCppVisitor
 * \brief %Visitor for printing C++ code compatible with legacy api of NEURON
 *
 * \todo
 *  - Handle define statement (i.e. macros)
 *  - If there is a return statement in the verbatim block
 *    of inlined function then it will be error. Need better
 *    error checking. For example, see netstim.mod where we
 *    have removed return from verbatim block.
 */
class CodegenNeuronCppVisitor: public CodegenCppVisitor {
  protected:
    using SymbolType = std::shared_ptr<symtab::Symbol>;

    /**
     * A vector of parameters represented by a 4-tuple of strings:
     *
     * - type qualifier (e.g. \c const)
     * - type (e.g. \c double)
     * - pointer qualifier (e.g. \c \_\_restrict\_\_)
     * - parameter name (e.g. \c data)
     *
     */
    using ParamVector = std::vector<std::tuple<std::string, std::string, std::string, std::string>>;


    /**
     * Add quotes to string to be output
     *
     * \param text The string to be quoted
     * \return     The same string with double-quotes pre- and postfixed
     */
    std::string add_escape_quote(const std::string& text) const {
        return "\"" + text + "\"";
    }

    /**
     * Data type for the local variables
     */
    const char* local_var_type() const noexcept {
        return codegen::naming::DEFAULT_LOCAL_VAR_TYPE;
    }


    /**
     * Default data type for floating point elements
     */
    const char* default_float_data_type() const noexcept {
        return codegen::naming::DEFAULT_FLOAT_TYPE;
    }


    /**
     * Data type for floating point elements specified on command line
     */
    const std::string& float_data_type() const noexcept {
        return float_type;
    }


    /**
     * Default data type for integer (offset) elements
     */
    const char* default_int_data_type() const noexcept {
        return codegen::naming::DEFAULT_INTEGER_TYPE;
    }


    /**
     * Constructs the name of a function or procedure
     * \param name The name of the function or procedure
     * \return     The name of the function or procedure postfixed with the model name
     */
    std::string method_name(const std::string& name) const {
        return name + "_" + info.mod_suffix;
    }


    /**
     * Creates a temporary symbol
     * \param name The name of the symbol
     * \return     A symbol based on the given name
     */
    SymbolType make_symbol(const std::string& name) const {
        return std::make_shared<symtab::Symbol>(name, ModToken());
    }


    /**
     * Check if nrn_state function is required
     */
    bool nrn_state_required() const noexcept;


    /**
     * Check if nrn_cur function is required
     */
    bool nrn_cur_required() const noexcept;


    /**
     * Check if net_receive function is required
     */
    bool net_receive_required() const noexcept;


    /**
     * Check if net_send_buffer is required
     */
    bool net_send_buffer_required() const noexcept;


    /**
     * Check if setup_range_variable function is required
     * \return
     */
    bool range_variable_setup_required() const noexcept;


    /**
     * Check if breakpoint node exist
     */
    bool breakpoint_exist() const noexcept;


    /**
     * Check if a semicolon is required at the end of given statement
     * \param node The AST Statement node to check
     * \return     \c true if this Statement requires a semicolon
     */
    static bool need_semicolon(const ast::Statement& node);


    /**
     * Number of float variables in the model
     */
    int float_variables_size() const;


    /**
     * Number of integer variables in the model
     */
    int int_variables_size() const;


    /**
     * Name of the simulator the code was generated for
     */
    std::string simulator_name() override;


    /**
     * Name of the code generation backend
     */
    virtual std::string backend_name() const;


    /**
     * Convert a given \c double value to its string representation
     * \param value The number to convert given as string as it is parsed by the modfile
     * \return      Its string representation
     */
    virtual std::string format_double_string(const std::string& value);


    /**
     * Convert a given \c float value to its string representation
     * \param value The number to convert given as string as it is parsed by the modfile
     * \return      Its string representation
     */
    virtual std::string format_float_string(const std::string& value);


    /**
     * Determine the name of a \c float variable given its symbol
     *
     * This function typically returns the accessor expression in backend code for the given symbol.
     * Since the model variables are stored in data arrays and accessed by offset, this function
     * will return the C++ string representing the array access at the correct offset
     *
     * \param symbol       The symbol of a variable for which we want to obtain its name
     * \param use_instance Should the variable be accessed via instance or data array
     * \return             The backend code string representing the access to the given variable
     * symbol
     */
    std::string float_variable_name(const SymbolType& symbol, bool use_instance) const override;


    /**
     * Determine the name of an \c int variable given its symbol
     *
     * This function typically returns the accessor expression in backend code for the given symbol.
     * Since the model variables are stored in data arrays and accessed by offset, this function
     * will return the C++ string representing the array access at the correct offset
     *
     * \param symbol       The symbol of a variable for which we want to obtain its name
     * \param name         The name of the index variable
     * \param use_instance Should the variable be accessed via instance or data array
     * \return             The backend code string representing the access to the given variable
     * symbol
     */
    std::string int_variable_name(const IndexVariableInfo& symbol,
                                  const std::string& name,
                                  bool use_instance) const override;


    /**
     * Determine the variable name for a global variable given its symbol
     * \param symbol The symbol of a variable for which we want to obtain its name
     * \param use_instance Should the variable be accessed via the (host-only)
     * global variable or the instance-specific copy (also available on GPU).
     * \return       The C++ string representing the access to the global variable
     */
    std::string global_variable_name(const SymbolType& symbol,
                                     bool use_instance = true) const override;


    /**
     * Determine variable name in the structure of mechanism properties
     *
     * \param name         Variable name that is being printed
     * \param use_instance Should the variable be accessed via instance or data array
     * \return             The C++ string representing the access to the variable in the neuron
     * thread structure
     */
    std::string get_variable_name(const std::string& name, bool use_instance = true) const override;


    /**
     * Determine the variable name for the "current" used in breakpoint block taking into account
     * intermediate code transformations.
     * \param current The variable name for the current used in the model
     * \return        The name for the current to be printed in C++
     */
    std::string breakpoint_current(std::string current) const;


    /**
     * populate all index semantics needed for registration with NEURON
     */
    void update_index_semantics();


    /**
     * Determine all \c float variables required during code generation
     * \return A \c vector of \c float variables
     */
    std::vector<SymbolType> get_float_variables() const;


    /**
     * Determine all \c int variables required during code generation
     * \return A \c vector of \c int variables
     */
    std::vector<IndexVariableInfo> get_int_variables();


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
     * For a given output block type, return statements for all read ion variables
     *
     * \param type The type of code block being generated
     * \return     A \c vector of strings representing the reading of ion variables
     */
    std::vector<std::string> ion_read_statements(BlockType type) const;


    /**
     * For a given output block type, return minimal statements for all read ion variables
     *
     * \param type The type of code block being generated
     * \return     A \c vector of strings representing the reading of ion variables
     */
    std::vector<std::string> ion_read_statements_optimized(BlockType type) const;


    /**
     * For a given output block type, return statements for writing back ion variables
     *
     * \param type The type of code block being generated
     * \return     A \c vector of strings representing the write-back of ion variables
     */
    std::vector<ShadowUseStatement> ion_write_statements(BlockType type);


    /**
     * Return ion variable name and corresponding ion read variable name
     * \param name The ion variable name
     * \return     The ion read variable name
     */
    static std::pair<std::string, std::string> read_ion_variable_name(const std::string& name);


    /**
     * Return ion variable name and corresponding ion write variable name
     * \param name The ion variable name
     * \return     The ion write variable name
     */
    static std::pair<std::string, std::string> write_ion_variable_name(const std::string& name);


    /**
     * Arguments for functions that are defined and used internally.
     * \return the method arguments
     */
    std::string internal_method_arguments();


    /**
     * Parameters for internally defined functions
     * \return the method parameters
     */
    ParamVector internal_method_parameters();


    /**
     * Arguments for external functions called from generated code
     * \return A string representing the arguments passed to an external function
     */
    static const char* external_method_arguments() noexcept;


    /**
     * Parameters for functions in generated code that are called back from external code
     *
     * Functions registered in NEURON during initialization for callback must adhere to a prescribed
     * calling convention. This method generates the string representing the function parameters for
     * these externally called functions.
     * \param table
     * \return      A string representing the parameters of the function
     */
    static const char* external_method_parameters(bool table = false) noexcept;


    /**
     * Arguments for register_mech or point_register_mech function
     */
    std::string register_mechanism_arguments() const;


    /**
     * Arguments for "_threadargs_" macro in neuron implementation
     */
    std::string nrn_thread_arguments() const override;


    /**
     * Arguments for "_threadargs_" macro in neuron implementation
     */
    std::string nrn_thread_internal_arguments();


    /**
     * Return the name of main compute kernels
     * \param type A block type
     */
    virtual std::string compute_method_name(BlockType type) const;


    /**
     * Process a verbatim block for possible variable renaming
     * \param text The verbatim code to be processed
     * \return     The code with all variables renamed as needed
     */
    std::string process_verbatim_text(std::string const& text) override;


    /**
     * Prints the start of the \c neuron namespace
     */
    void print_namespace_start();


    /**
     * Prints the end of the \c neuron namespace
     */
    void print_namespace_stop();


    /**
     * Print the nmodl constants used in backend code
     *
     * Currently we define three basic constants, which are assumed to be present in NMODL, directly
     * in the backend code:
     *
     * \code
     * static const double FARADAY = 96485.3;
     * static const double PI = 3.14159;
     * static const double R = 8.3145;
     * \endcode
     */
    virtual void print_nmodl_constants();


    /**
     * Print top file header printed in generated code
     */
    void print_backend_info();


    /**
     * Print memory allocation routine
     */
    virtual void print_memory_allocation_routine() const;


    /**
     * Print backend specific abort routine
     */
    virtual void print_abort_routine() const;


    /**
     * Print standard C/C++ includes
     */
    void print_standard_includes();


    /**
     * Print includes from NEURON
     */
    void print_neuron_includes();


    /**
     * Print declaration of macro NRN_PRCELLSTATE for debugging
     */
    void print_prcellstate_macros() const;

    /**
     * Print backend code for byte array that has mechanism information (to be registered
     * with NEURON)
     */
    void print_mechanism_info();


    /**
     * Print the structure that wraps all global variables used in the NMODL
     *
     * \param print_initializers Whether to include default values in the struct
     *                           definition (true: int foo{42}; false: int foo;)
     */
    void print_mechanism_global_var_structure(bool print_initializers);


    /**
     * Print byte arrays that register scalar and vector variables for hoc interface
     *
     */
    void print_global_variables_for_hoc();


    /**
     * Print atomic update pragma for reduction statements
     */
    virtual void print_atomic_reduction_pragma() override;


    /**
     * Print call to internal or external function
     * \param node The AST node representing a function call
     */
    void print_function_call(const ast::FunctionCall& node) override;


    /**
     * Print function and procedures prototype declaration
     */
    void print_function_prototypes();


    /**
     * Print nmodl function or procedure (common code)
     * \param node the AST node representing the function or procedure in NMODL
     * \param name the name of the function or procedure
     */
    void print_function_or_procedure(const ast::Block& node, const std::string& name);


    /**
     * Common helper function to help printing function or procedure blocks
     * \param node the AST node representing the function or procedure in NMODL
     */
    void print_function_procedure_helper(const ast::Block& node);


    /**
     * Print prototype declarations of functions or procedures
     * \tparam T   The AST node type of the node (must be of nmodl::ast::Ast or subclass)
     * \param node The AST node representing the function or procedure block
     * \param name A user defined name for the function
     */
    template <typename T>
    void print_function_declaration(const T& node, const std::string& name);


    /**
     * Print nrn_constructor function definition
     *
     */
    void print_nrn_constructor();


    /**
     * Print nrn_destructor function definition
     *
     */
    void print_nrn_destructor();


    /**
     * Print nrn_alloc function definition
     *
     */
    void print_nrn_alloc();


    /**
     * Print common code for global functions like nrn_init, nrn_cur and nrn_state
     * \param type The target backend code block type
     */
    virtual void print_global_function_common_code(BlockType type,
                                                   const std::string& function_name = "");


    /**
     * Print the mechanism registration function
     *
     */
    void print_mechanism_register();


    /**
     * Print all includes
     *
     */
    void print_headers_include() override;


    /**
     * Print all NEURON macros
     *
     */
    void print_macro_definitions();


    /**
     * Print NEURON global variable macros
     *
     */
    void print_global_macros();


    /**
     * Print mechanism variables' related macros
     *
     */
    void print_mechanism_variables_macros();


    /**
     * Print start of namespaces
     *
     */
    void print_namespace_begin();


    /**
     * Print end of namespaces
     *
     */
    void print_namespace_end();


    /**
     * Print all classes
     * \param print_initializers Whether to include default values.
     */
    void print_data_structures(bool print_initializers);


    /**
     * Set v_unused (voltage) for NRN_PRCELLSTATE feature
     */
    void print_v_unused() const;


    /**
     * Set g_unused (conductance) for NRN_PRCELLSTATE feature
     */
    void print_g_unused() const;


    /**
     * Print all compute functions for every backend
     *
     */
    virtual void print_compute_functions();


    /**
     * Print entry point to code generation
     *
     */
    virtual void print_codegen_routines();


  public:
    /**
     * \brief Constructs the C++ code generator visitor
     *
     * This constructor instantiates an NMODL C++ code generator and allows writing generated code
     * directly to a file in \c [output_dir]/[mod_filename].cpp.
     *
     * \note No code generation is performed at this stage. Since the code
     * generator classes are all based on \c AstVisitor the AST must be visited using e.g. \c
     * visit_program in order to generate the C++ code corresponding to the AST.
     *
     * \param mod_filename The name of the model for which code should be generated.
     *                     It is used for constructing an output filename.
     * \param output_dir   The directory where target C++ file should be generated.
     * \param float_type   The float type to use in the generated code. The string will be used
     *                     as-is in the target code. This defaults to \c double.
     */
    CodegenNeuronCppVisitor(std::string mod_filename,
                            const std::string& output_dir,
                            std::string float_type,
                            const bool optimize_ionvar_copies)
        : CodegenCppVisitor(mod_filename, output_dir, float_type, optimize_ionvar_copies) {}

    /**
     * \copybrief nmodl::codegen::CodegenNeuronCppVisitor
     *
     * This constructor instantiates an NMODL C++ code generator and allows writing generated code
     * into an output stream.
     *
     * \note No code generation is performed at this stage. Since the code
     * generator classes are all based on \c AstVisitor the AST must be visited using e.g. \c
     * visit_program in order to generate the C++ code corresponding to the AST.
     *
     * \param mod_filename The name of the model for which code should be generated.
     *                     It is used for constructing an output filename.
     * \param stream       The output stream onto which to write the generated code
     * \param float_type   The float type to use in the generated code. The string will be used
     *                     as-is in the target code. This defaults to \c double.
     */
    CodegenNeuronCppVisitor(std::string mod_filename,
                            std::ostream& stream,
                            std::string float_type,
                            const bool optimize_ionvar_copies)
        : CodegenCppVisitor(mod_filename, stream, float_type, optimize_ionvar_copies) {}

    /**
     * Main and only member function to call after creating an instance of this class.
     * \param program the AST to translate to C++ code
     */
    void visit_program(const ast::Program& program) override;


    /**
     * Print nrn_state / state update function definition
     */
    void print_nrn_state() override;


    /**
     * Print NMODL function in target backend code
     * \param node
     */
    void print_function(const ast::FunctionBlock& node);


    /**
     * Print NMODL procedure in target backend code
     * \param node
     */
    virtual void print_procedure(const ast::ProcedureBlock& node);


    /** Setup the target backend code generator
     *
     * Typically called from within \c visit\_program but may be called from
     * specialized targets to setup this Code generator as fallback.
     */
    void setup(const ast::Program& node);


    /**
     * Print the structure that wraps all range and int variables required for the NMODL
     *
     * \param print_initializers Whether or not default values for variables
     *                           be included in the struct declaration.
     */
    void print_mechanism_range_var_structure(bool print_initializers);


    /**
     * \brief Based on the \c EigenNewtonSolverBlock passed print the definition needed for its
     * functor
     *
     * \param node \c EigenNewtonSolverBlock for which to print the functor
     */
    void print_functor_definition(const ast::EigenNewtonSolverBlock& node);

    virtual void visit_solution_expression(const ast::SolutionExpression& node) override;
    virtual void visit_watch_statement(const ast::WatchStatement& node) override;
};


template <typename T>
void CodegenNeuronCppVisitor::print_vector_elements(const std::vector<T>& elements,
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


/**
 * \details If there is an argument with name (say alpha) same as range variable (say alpha),
 * we want to avoid it being printed as instance->alpha. And hence we disable variable
 * name lookup during prototype declaration. Note that the name of procedure can be
 * different in case of table statement.
 */
template <typename T>
void CodegenNeuronCppVisitor::print_function_declaration(const T& node, const std::string& name) {
    enable_variable_name_lookup = false;
    auto type = default_float_data_type();

    // internal and user provided arguments
    auto internal_params = internal_method_parameters();
    const auto& params = node.get_parameters();
    for (const auto& param: params) {
        internal_params.emplace_back("", type, "", param.get()->get_node_name());
    }

    // procedures have "int" return type by default
    const char* return_type = "int";
    if (node.is_function_block()) {
        return_type = default_float_data_type();
    }

    printer->add_indent();
    printer->fmt_text("inline {} {}({})", return_type, method_name(name), "params");

    enable_variable_name_lookup = true;
}

/** \} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl

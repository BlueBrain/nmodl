/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "codegen/codegen_cpp_visitor.hpp"

/// encapsulates code generation backend implementations
namespace nmodl {

namespace codegen {

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
class CodegenNrnCppVisitor: public visitor::CodegenCppVisitor {
  protected:
    /**
     * Name of the code generation backend
     */
    std::string backend_name() const;


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
                                  bool use_instance) const;


    /**
     * Determine the variable name for a global variable given its symbol
     * \param symbol The symbol of a variable for which we want to obtain its name
     * \param use_instance Should the variable be accessed via the (host-only)
     * global variable or the instance-specific copy (also available on GPU).
     * \return       The C++ string representing the access to the global variable
     */
    std::string global_variable_name(const SymbolType& symbol, bool use_instance = true) const;


    /**
     * Determine variable name in the structure of mechanism properties
     *
     * \param name         Variable name that is being printed
     * \param use_instance Should the variable be accessed via instance or data array
     * \return             The C++ string representing the access to the variable in the neuron
     * thread structure
     */
    std::string get_variable_name(const std::string& name, bool use_instance = true) const;

    /**
     * Generate Function call statement for nrn_wrote_conc
     * \param ion_name      The name of the ion variable
     * \param concentration The name of the concentration variable
     * \param index
     * \return              The string representing the function call
     */
    std::string conc_write_statement(const std::string& ion_name,
                                     const std::string& concentration,
                                     int index);


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
    const char* external_method_arguments() noexcept;


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
    std::string nrn_thread_arguments() const;


    /**
     * Arguments for "_threadargs_" macro in neuron implementation
     */
    std::string nrn_thread_internal_arguments();


     /**
     * Return the name of main compute kernels
     * \param type A block type
     */
    std::string compute_method_name(BlockType type) const;


    /**
     * The used global type qualifier
     *
     * For C++ code generation this is empty
     * \return ""
     *
     * \return "uniform "
     */
    std::string global_var_struct_type_qualifier();

    /**
     * Instantiate global var instance
     *
     * For C++ code generation this is empty
     * \return ""
     */
    void print_global_var_struct_decl();

    /**
     * Print static assertions about the global variable struct.
     */
    void print_global_var_struct_assertions() const;

    /**
     * Prints the start of the \c coreneuron namespace
     */
    void print_namespace_start();


    /**
     * Prints the end of the \c coreneuron namespace
     */
    void print_namespace_stop();


    /**
     * Prints the start of namespace for the backend-specific code
     *
     * For the C++ backend no additional namespace is required
     */
    void print_backend_namespace_start();


    /**
     * Prints the end of namespace for the backend-specific code
     *
     * For the C++ backend no additional namespace is required
     */
    void print_backend_namespace_stop();


    std::string CodegenCppVisitor::simulator_name() const;


    /**
     * Print backend specific abort routine
     */
    void print_abort_routine() const;


    /**
     * Print standard C/C++ includes
     */
    void print_standard_includes();


    /**
     * Print includes from coreneuron
     */
    void print_simulator_includes();


    /**
     * Print backend specific includes (none needed for C++ backend)
     */
    void print_backend_includes();


    /**
     * Check if reduction block in \c nrn\_cur required
     */
    bool nrn_cur_reduction_loop_required();


    /**
     * Print check_table functions
     */
    // Different in both
    void print_check_table_thread_function();


    /**
     * Print top level (global scope) verbatim blocks
     */
    // Bit different in both
    void print_top_verbatim_blocks();


    /**
     * Print initial block statements
     *
     * Generate the target backend code corresponding to the NMODL initial block statements
     *
     * \param node The AST Node representing a NMODL initial block
     */
    void print_initial_block(const ast::InitialBlock* node);


    /**
     * Print initial block in the net receive block
     */
    void print_net_init();


    /**
     * Print the common code section for net receive related methods
     *
     * \param node The AST node representing the corresponding NMODL block
     * \param need_mech_inst \c true if a local \c inst variable needs to be defined in generated
     * code
     */
    void print_net_receive_common_code(const ast::Block& node, bool need_mech_inst = true);


    /**
     * Print the code related to the update of NetSendBuffer_t cnt. For GPU this needs to be done
     * with atomic operation, on CPU it's not needed.
     *
     */
    void print_net_send_buffering_cnt_update() const;


    /**
     * Print statement that grows NetSendBuffering_t structure if needed.
     * This function should be overridden for backends that cannot dynamically reallocate the buffer
     */
    void print_net_send_buffering_grow();


    /**
     * Print kernel for buffering net_send events
     *
     * This kernel is only needed for accelerator backends where \c net\_send needs to be executed
     * in two stages as the actual communication must be done in the host code.
     */
    void print_net_send_buffering();


    /**
     * Print send event move block used in net receive as well as watch
     */
    void print_send_event_move();


    /**
     * Generate the target backend code for the \c net\_receive\_buffering function delcaration
     * \return The target code string
     */
    std::string net_receive_buffering_declaration();


    /**
     * Print the target backend code for defining and checking a local \c Memb\_list variable
     */
    void print_get_memb_list();


    /**
     * Print the code for the main \c net\_receive loop
     */
    void print_net_receive_loop_begin();


    /**
     * Print the code for closing the main \c net\_receive loop
     */
    void print_net_receive_loop_end();


    /**
     * Print \c net\_receive function definition
     */
    void print_net_receive();


    /**
     * Print derivative kernel when \c derivimplicit method is used
     *
     * \param block The corresponding AST node representing an NMODL \c derivimplicit block
     */
    void print_derivimplicit_kernel(const ast::Block& block);


    /**
     * Print main body of nrn_cur function
     * \param node the AST node representing the NMODL breakpoint block
     */
    void print_nrn_cur_kernel(const ast::BreakpointBlock& node);


    /**
     * Print the \c nrn\_cur kernel with NMODL \c conductance keyword provisions
     *
     * If the NMODL \c conductance keyword is used in the \c breakpoint block, then
     * CodegenCppVisitor::print_nrn_cur_kernel will use this printer
     *
     * \param node the AST node representing the NMODL breakpoint block
     */
    void print_nrn_cur_conductance_kernel(const ast::BreakpointBlock& node);


    /**
     * Print the \c nrn\_cur kernel without NMODL \c conductance keyword provisions
     *
     * If the NMODL \c conductance keyword is \b not used in the \c breakpoint block, then
     * CodegenCppVisitor::print_nrn_cur_kernel will use this printer
     */
    void print_nrn_cur_non_conductance_kernel();


    /**
     * Print the \c nrn_current kernel
     *
     * \note nrn_cur_kernel will have two calls to nrn_current if no conductance keywords specified
     * \param node the AST node representing the NMODL breakpoint block
     */
    void print_nrn_current(const ast::BreakpointBlock& node);


    /**
     * Print the update to matrix elements with/without shadow vectors
     *
     */
    void print_nrn_cur_matrix_shadow_update();


    /**
     * Print the reduction to matrix elements from shadow vectors
     *
     */
    void print_nrn_cur_matrix_shadow_reduction();


    /**
     * Print nrn_alloc function definition
     *
     */
    // NEURON only
    void print_nrn_alloc();


    /**
     * Print common code for global functions like nrn_init, nrn_cur and nrn_state
     * \param type The target backend code block type
     */
    void print_global_function_common_code(BlockType type,
                                                   const std::string& function_name = "");


    /**
     * Print the mechanism registration function
     *
     */
    void print_mechanism_register();


    /**
     * Print watch activate function
     *
     */
    void print_watch_activate();


    /**
     * Print all includes
     *
     */
    void print_headers_include();


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
     * Print common getters
     *
     */
    void print_common_getters();


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
    void print_compute_functions();


    /**
     * Print entry point to code generation
     *
     */
    void print_codegen_routines();


    /// This constructor is private, see the public section below to find how to create an instance
    /// of this class.
    CodegenCppVisitor(std::string mod_filename,
                      const std::string& output_dir,
                      std::string float_type,
                      const bool optimize_ionvar_copies,
                      const std::string& extension,
                      const std::string& wrapper_ext)
        : printer(std::make_unique<CodePrinter>(output_dir + "/" + mod_filename + extension))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}

    /// This constructor is private, see the public section below to find how to create an instance
    /// of this class.
    CodegenCppVisitor(std::string mod_filename,
                      std::ostream& stream,
                      std::string float_type,
                      const bool optimize_ionvar_copies,
                      const std::string& /* extension */,
                      const std::string& /* wrapper_ext */)
        : printer(std::make_unique<CodePrinter>(stream))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}


  public:
    /**
     * \brief Constructs the C++ code generator visitor
     *
     * This constructor instantiates an NMODL C++ code generator and allows writing generated code
     * directly to a file in \c [output_dir]/[mod_filename].[extension].
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
     * \param extension    The file extension to use. This defaults to \c .cpp .
     */
    CodegenCppVisitor(std::string mod_filename,
                      const std::string& output_dir,
                      std::string float_type,
                      const bool optimize_ionvar_copies,
                      const std::string& extension = ".cpp")
        : printer(std::make_unique<CodePrinter>(output_dir + "/" + mod_filename + extension))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}

    /**
     * \copybrief nmodl::codegen::CodegenCppVisitor
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
    CodegenCppVisitor(std::string mod_filename,
                      std::ostream& stream,
                      std::string float_type,
                      const bool optimize_ionvar_copies)
        : printer(std::make_unique<CodePrinter>(stream))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}

    /**
     * Main and only member function to call after creating an instance of this class.
     * \param program the AST to translate to C++ code
     */
    void visit_program(const ast::Program& program) override;

    /**
     * Print the \c nrn\_init function definition
     * \param skip_init_check \c true to generate code executing the initialization conditionally
     */
    void print_nrn_init(bool skip_init_check = true);


    /**
     * Print nrn_state / state update function definition
     */
    void print_nrn_state();


    /**
     * Print nrn_cur / current update function definition
     */
    void print_nrn_cur();

    /**
     * Print fast membrane current calculation code
     */
    void print_fast_imem_calculation();


    /**
     * Print kernel for buffering net_receive events
     *
     * This kernel is only needed for accelerator backends where \c net\_receive needs to be
     * executed in two stages as the actual communication must be done in the host code. \param
     * need_mech_inst \c true if the generated code needs a local inst variable to be defined
     */
    void print_net_receive_buffering(bool need_mech_inst = true);


    /**
     * Print \c net\_receive kernel function definition
     */
    void print_net_receive_kernel();


    /**
     * Print watch activate function
     */
    void print_watch_check();


    /**
     * Print \c check\_function() for functions or procedure using table
     * \param node The AST node representing a function or procedure block
     */
    // Probably different between NEURON and CoreNEURON but shouldn't be too different
    void print_table_check_function(const ast::Block& node);


    /**
     * Print replacement function for function or procedure using table
     * \param node The AST node representing a function or procedure block
     */
    // Probably different between NEURON and CoreNEURON but shouldn't be too different
    void print_table_replacement_function(const ast::Block& node);


    /**
     * Print NMODL function_table in target backend code
     * \param node
     */
    void print_function_tables(const ast::FunctionTableBlock& node);


    /**
     * Print NMODL procedure in target backend code
     * \param node
     */
    void print_procedure(const ast::ProcedureBlock& node);

    /**
     * Print NMODL before / after block in target backend code
     * \param node AST node of type before/after type being printed
     * \param block_id Index of the before/after block
     */
    void print_before_after_block(const ast::Block* node, size_t block_id);

    /** Setup the target backend code generator
     *
     * Typically called from within \c visit\_program but may be called from
     * specialized targets to setup this Code generator as fallback.
     */
    void setup(const ast::Program& node);


    /**
     * Set the global variables to be generated in target backend code
     * \param global_vars
     */
    void set_codegen_global_variables(const std::vector<SymbolType>& global_vars);

    /**
     * Find unique variable name defined in nmodl::utils::SingletonRandomString by the
     * nmodl::visitor::SympySolverVisitor
     * \param original_name Original name of variable to change
     * \return std::string Unique name produced as [original_name]_[random_string]
     */
    std::string find_var_unique_name(const std::string& original_name) const;

    /**
     * Print the structure that wraps all range and int variables required for the NMODL
     *
     * \param print_initializers Whether or not default values for variables
     *                           be included in the struct declaration.
     */
    void print_mechanism_range_var_structure(bool print_initializers);

    /**
     * Print the function that initialize instance structure
     */
    void print_instance_variable_setup();

    /**
     * Go through the map of \c EigenNewtonSolverBlock s and their corresponding functor names
     * and print the functor definitions before the definitions of the functions of the generated
     * file
     *
     */
    void print_functors_definitions();

    /**
     * \brief Based on the \c EigenNewtonSolverBlock passed print the definition needed for its
     * functor
     *
     * \param node \c EigenNewtonSolverBlock for which to print the functor
     */
    void print_functor_definition(const ast::EigenNewtonSolverBlock& node);

};

/** \} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl

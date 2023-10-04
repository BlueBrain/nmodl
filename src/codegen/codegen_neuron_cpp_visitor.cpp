/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "codegen/codegen_neuron_cpp_visitor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <regex>

#include "ast/all.hpp"
#include "codegen/codegen_helper_visitor.hpp"
#include "codegen/codegen_naming.hpp"
#include "codegen/codegen_utils.hpp"
#include "config/config.h"
#include "lexer/token_mapping.hpp"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"
#include "utils/string_utils.hpp"
#include "visitors/defuse_analyze_visitor.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/var_usage_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

using visitor::DefUseAnalyzeVisitor;
using visitor::DUState;
using visitor::RenameVisitor;
using visitor::SymtabVisitor;
using visitor::VarUsageVisitor;

using symtab::syminfo::NmodlType;


/****************************************************************************************/
/*                               Common helper routines                                 */
/****************************************************************************************/


/**
 * \details Current variable used in breakpoint block could be local variable.
 * In this case, neuron has already renamed the variable name by prepending
 * "_l". In our implementation, the variable could have been renamed by
 * one of the pass. And hence, we search all local variables and check if
 * the variable is renamed. Note that we have to look into the symbol table
 * of statement block and not breakpoint.
 */
std::string CodegenNeuronCppVisitor::breakpoint_current(std::string current) const {
    auto breakpoint = info.breakpoint_node;
    if (breakpoint == nullptr) {
        return current;
    }
    auto symtab = breakpoint->get_statement_block()->get_symbol_table();
    auto variables = symtab->get_variables_with_properties(NmodlType::local_var);
    for (const auto& var: variables) {
        auto renamed_name = var->get_name();
        auto original_name = var->get_original_name();
        if (current == original_name) {
            current = renamed_name;
            break;
        }
    }
    return current;
}


int CodegenNeuronCppVisitor::float_variables_size() const {
    return codegen_float_variables.size();
}


int CodegenNeuronCppVisitor::int_variables_size() const {
    const auto count_semantics = [](int sum, const IndexSemantics& sem) { return sum += sem.size; };
    return std::accumulate(info.semantics.begin(), info.semantics.end(), 0, count_semantics);
}

std::pair<std::string, std::string> CodegenNeuronCppVisitor::read_ion_variable_name(
    const std::string& name) {
    return {name, naming::ION_VARNAME_PREFIX + name};
}


std::pair<std::string, std::string> CodegenNeuronCppVisitor::write_ion_variable_name(
    const std::string& name) {
    return {naming::ION_VARNAME_PREFIX + name, name};
}

/**
 * \details Depending upon the block type, we have to print read/write ion variables
 * during code generation. Depending on block/procedure being printed, this
 * method return statements as vector. As different code backends could have
 * different variable names, we rely on backend-specific read_ion_variable_name
 * and write_ion_variable_name method which will be overloaded.
 */
std::vector<std::string> CodegenNeuronCppVisitor::ion_read_statements(BlockType type) const {
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        auto name = ion.name;
        for (const auto& var: ion.reads) {
            auto const iter = std::find(ion.implicit_reads.begin(), ion.implicit_reads.end(), var);
            if (iter != ion.implicit_reads.end()) {
                continue;
            }
            auto variable_names = read_ion_variable_name(var);
            auto first = get_variable_name(variable_names.first);
            auto second = get_variable_name(variable_names.second);
            statements.push_back(fmt::format("{} = {};", first, second));
        }
        for (const auto& var: ion.writes) {
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = get_variable_name(variables.first);
                auto second = get_variable_name(variables.second);
                statements.push_back(fmt::format("{} = {};", first, second));
            }
        }
    }
    return statements;
}


std::vector<std::string> CodegenNeuronCppVisitor::ion_read_statements_optimized(
    BlockType type) const {
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        for (const auto& var: ion.writes) {
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = "ionvar." + variables.first;
                const auto& second = get_variable_name(variables.second);
                statements.push_back(fmt::format("{} = {};", first, second));
            }
        }
    }
    return statements;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<ShadowUseStatement> CodegenNeuronCppVisitor::ion_write_statements(BlockType type) {
    std::vector<ShadowUseStatement> statements;
    for (const auto& ion: info.ions) {
        std::string concentration;
        auto name = ion.name;
        for (const auto& var: ion.writes) {
            auto variable_names = write_ion_variable_name(var);
            if (ion.is_ionic_current(var)) {
                if (type == BlockType::Equation) {
                    auto current = breakpoint_current(var);
                    auto lhs = variable_names.first;
                    auto op = "+=";
                    auto rhs = get_variable_name(current);
                    if (info.point_process) {
                        auto area = get_variable_name(naming::NODE_AREA_VARIABLE);
                        rhs += fmt::format("*(1.e2/{})", area);
                    }
                    statements.push_back(ShadowUseStatement{lhs, op, rhs});
                }
            } else {
                if (!ion.is_rev_potential(var)) {
                    concentration = var;
                }
                auto lhs = variable_names.first;
                auto op = "=";
                auto rhs = get_variable_name(variable_names.second);
                statements.push_back(ShadowUseStatement{lhs, op, rhs});
            }
        }

        if (type == BlockType::Initial && !concentration.empty()) {
            int index = 0;
            if (ion.is_intra_cell_conc(concentration)) {
                index = 1;
            } else if (ion.is_extra_cell_conc(concentration)) {
                index = 2;
            } else {
                /// \todo Unhandled case in neuron implementation
                throw std::logic_error(fmt::format("codegen error for {} ion", ion.name));
            }
            auto ion_type_name = fmt::format("{}_type", ion.name);
            auto lhs = fmt::format("int {}", ion_type_name);
            auto op = "=";
            auto rhs = get_variable_name(ion_type_name);
            statements.push_back(ShadowUseStatement{lhs, op, rhs});
            // auto statement = conc_write_statement(ion.name, concentration, index);
            // statements.push_back(ShadowUseStatement{statement, "", ""});
        }
    }
    return statements;
}


/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/


std::string CodegenNeuronCppVisitor::simulator_name() {
    return "NEURON";
}


std::string CodegenNeuronCppVisitor::backend_name() const {
    return "C++ (api-compatibility)";
}


void CodegenNeuronCppVisitor::print_memory_allocation_routine() const {
    printer->add_newline(2);
    auto args = "size_t num, size_t size, size_t alignment = 16";
    printer->fmt_push_block("static inline void* mem_alloc({})", args);
    printer->add_line("void* ptr;");
    printer->add_line("posix_memalign(&ptr, alignment, num*size);");
    printer->add_line("memset(ptr, 0, size);");
    printer->add_line("return ptr;");
    printer->pop_block();

    printer->add_newline(2);
    printer->push_block("static inline void mem_free(void* ptr)");
    printer->add_line("free(ptr);");
    printer->pop_block();
}

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_abort_routine() const {
    printer->add_newline(2);
    printer->push_block("static inline void coreneuron_abort()");
    printer->add_line("abort();");
    printer->pop_block();
}


std::string CodegenNeuronCppVisitor::compute_method_name(BlockType type) const {
    if (type == BlockType::Initial) {
        return method_name(naming::NRN_INIT_METHOD);
    }
    if (type == BlockType::Constructor) {
        return method_name(naming::NRN_CONSTRUCTOR_METHOD);
    }
    if (type == BlockType::Destructor) {
        return method_name(naming::NRN_DESTRUCTOR_METHOD);
    }
    if (type == BlockType::State) {
        return method_name(naming::NRN_STATE_METHOD);
    }
    if (type == BlockType::Equation) {
        return method_name(naming::NRN_CUR_METHOD);
    }
    if (type == BlockType::Watch) {
        return method_name(naming::NRN_WATCH_CHECK_METHOD);
    }
    throw std::logic_error("compute_method_name not implemented");
}


/****************************************************************************************/
/*              printing routines for code generation                                   */
/****************************************************************************************/

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::visit_watch_statement(const ast::WatchStatement& /* node */) {
    return;
}


// TODO: Check what we do in NEURON
void CodegenNeuronCppVisitor::print_atomic_reduction_pragma() {
    return;
}

void CodegenNeuronCppVisitor::print_statement_block(const ast::StatementBlock& node,
                                                    bool open_brace,
                                                    bool close_brace) {
    if (open_brace) {
        printer->push_block();
    }

    const auto& statements = node.get_statements();
    for (const auto& statement: statements) {
        if (statement_to_skip(*statement)) {
            continue;
        }
        /// not necessary to add indent for verbatim block (pretty-printing)
        if (!statement->is_verbatim() && !statement->is_mutex_lock() &&
            !statement->is_mutex_unlock() && !statement->is_protect_statement()) {
            printer->add_indent();
        }
        statement->accept(*this);
        if (need_semicolon(*statement)) {
            printer->add_text(';');
        }
        if (!statement->is_mutex_lock() && !statement->is_mutex_unlock()) {
            printer->add_newline();
        }
    }

    if (close_brace) {
        printer->pop_block_nl(0);
    }
}

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_function_call(const FunctionCall& node) {
    const auto& name = node.get_node_name();
    auto function_name = name;
    // if (defined_method(name)) {
    //     function_name = method_name(name);
    // }

    // if (is_net_send(name)) {
    //     print_net_send_call(node);
    //     return;
    // }

    // if (is_net_move(name)) {
    //     print_net_move_call(node);
    //     return;
    // }

    // if (is_net_event(name)) {
    //     print_net_event_call(node);
    //     return;
    // }

    const auto& arguments = node.get_arguments();
    printer->add_text(function_name, '(');

    // if (defined_method(name)) {
    //     printer->add_text(internal_method_arguments());
    //     if (!arguments.empty()) {
    //         printer->add_text(", ");
    //     }
    // }

    print_vector_elements(arguments, ", ");
    printer->add_text(')');
}


void CodegenNeuronCppVisitor::print_function_prototypes() {
    if (info.functions.empty() && info.procedures.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    for (const auto& node: info.functions) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(';');
        printer->add_newline();
    }
    for (const auto& node: info.procedures) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(';');
        printer->add_newline();
    }
    codegen = false;
}


// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_function_or_procedure(const ast::Block& node,
                                                          const std::string& name) {
    printer->add_newline(2);
    print_function_declaration(node, name);
    printer->add_text(" ");
    printer->push_block();

    // function requires return variable declaration
    if (node.is_function_block()) {
        auto type = default_float_data_type();
        printer->fmt_line("{} ret_{} = 0.0;", type, name);
    } else {
        printer->fmt_line("int ret_{} = 0;", name);
    }

    print_statement_block(*node.get_statement_block(), false, false);
    printer->fmt_line("return ret_{};", name);
    printer->pop_block();
}

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_function_procedure_helper(const ast::Block& node) {
    codegen = true;
    auto name = node.get_node_name();

    if (info.function_uses_table(name)) {
        auto new_name = "f_" + name;
        print_function_or_procedure(node, new_name);
    } else {
        print_function_or_procedure(node, name);
    }

    codegen = false;
}

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_procedure(const ast::ProcedureBlock& node) {
    print_function_procedure_helper(node);
}

// TODO: Edit for NEURON
void CodegenNeuronCppVisitor::print_function(const ast::FunctionBlock& node) {
    auto name = node.get_node_name();

    // name of return variable
    std::string return_var;
    if (info.function_uses_table(name)) {
        return_var = "ret_f_" + name;
    } else {
        return_var = "ret_" + name;
    }

    // first rename return variable name
    auto block = node.get_statement_block().get();
    RenameVisitor v(name, return_var);
    block->accept(v);

    print_function_procedure_helper(node);
}


/****************************************************************************************/
/*                           Code-specific helper routines                              */
/****************************************************************************************/


std::string CodegenNeuronCppVisitor::internal_method_arguments() {
    // TODO: rewrite based on NEURON
    return {};
}


/**
 * @todo: figure out how to correctly handle qualifiers
 */
CodegenNeuronCppVisitor::ParamVector CodegenNeuronCppVisitor::internal_method_parameters() {
    // TODO: rewrite based on NEURON
    return {};
}


const char* CodegenNeuronCppVisitor::external_method_arguments() noexcept {
    // TODO: rewrite based on NEURON
    return {};
}


const char* CodegenNeuronCppVisitor::external_method_parameters(bool table) noexcept {
    // TODO: rewrite based on NEURON
    return {};
}


std::string CodegenNeuronCppVisitor::nrn_thread_arguments() const {
    // TODO: rewrite based on NEURON
    return {};
}


/**
 * Function call arguments when function or procedure is defined in the
 * same mod file itself
 */
std::string CodegenNeuronCppVisitor::nrn_thread_internal_arguments() {
    // TODO: rewrite based on NEURON
    return {};
}


// TODO: Write for NEURON
std::string CodegenNeuronCppVisitor::process_verbatim_text(std::string const& text) {
    return {};
}


/****************************************************************************************/
/*               Code-specific printing routines for code generation                    */
/****************************************************************************************/


/**
 * NMODL constants from unit database
 *
 */
void CodegenNeuronCppVisitor::print_nmodl_constants() {
    if (!info.factor_definitions.empty()) {
        printer->add_newline(2);
        printer->add_line("/** constants used in nmodl from UNITS */");
        for (const auto& it: info.factor_definitions) {
            const std::string format_string = "static const double {} = {};";
            printer->fmt_line(format_string, it->get_node_name(), it->get_value()->get_value());
        }
    }
}


void CodegenNeuronCppVisitor::print_namespace_start() {
    printer->add_newline(2);
    printer->push_block("namespace neuron");
}


void CodegenNeuronCppVisitor::print_namespace_stop() {
    printer->pop_block();
}


/****************************************************************************************/
/*                         Routines for returning variable name                         */
/****************************************************************************************/


std::string CodegenNeuronCppVisitor::float_variable_name(const SymbolType& symbol,
                                                         bool use_instance) const {
    // TODO: rewrite for NEURON
    return symbol->get_name();
}


std::string CodegenNeuronCppVisitor::int_variable_name(const IndexVariableInfo& symbol,
                                                       const std::string& name,
                                                       bool use_instance) const {
    // TODO: rewrite for NEURON
    return name;
}


std::string CodegenNeuronCppVisitor::global_variable_name(const SymbolType& symbol,
                                                          bool use_instance) const {
    // TODO: rewrite for NEURON
    return symbol->get_name();
}


std::string CodegenNeuronCppVisitor::get_variable_name(const std::string& name,
                                                       bool use_instance) const {
    // TODO: rewrite for NEURON
    return name;
}


/****************************************************************************************/
/*                      Main printing routines for code generation                      */
/****************************************************************************************/


void CodegenNeuronCppVisitor::print_backend_info() {
    time_t current_time{};
    time(&current_time);
    std::string data_time_str{std::ctime(&current_time)};
    auto version = nmodl::Version::NMODL_VERSION + " [" + nmodl::Version::GIT_REVISION + "]";

    printer->add_line("/*********************************************************");
    printer->add_line("Model Name      : ", info.mod_suffix);
    printer->add_line("Filename        : ", info.mod_file, ".mod");
    printer->add_line("NMODL Version   : ", nmodl_version());
    printer->fmt_line("Vectorized      : {}", info.vectorize);
    printer->fmt_line("Threadsafe      : {}", info.thread_safe);
    printer->add_line("Created         : ", stringutils::trim(data_time_str));
    printer->add_line("Simulator       : ", simulator_name());
    printer->add_line("Backend         : ", backend_name());
    printer->add_line("NMODL Compiler  : ", version);
    printer->add_line("*********************************************************/");
}


void CodegenNeuronCppVisitor::print_standard_includes() {
    printer->add_newline();
    printer->add_multi_line(R"CODE(
        #include <math.h>
        #include <stdio.h>
        #include <stdlib.h>
    )CODE");
    if (!info.vectorize) {
        printer->add_line("#include <vector>");
    }
}


void CodegenNeuronCppVisitor::print_neuron_includes() {
    printer->add_newline();
    printer->add_multi_line(R"CODE(
        #include "mech_api.h"
        #include "neuron/cache/mechanism_range.hpp"
        #include "nrniv_mf.h"
        #include "section_fwd.hpp"
    )CODE");
}


void CodegenNeuronCppVisitor::print_global_macros() {
    printer->add_newline();
    printer->add_line("/* NEURON global macro definitions */");
    if (info.vectorize) {
        printer->add_multi_line(R"CODE(
            /* VECTORIZED */
            #define NRN_VECTORIZED 1
        )CODE");
    } else {
        printer->add_multi_line(R"CODE(
            /* NOT VECTORIZED */
            #define NRN_VECTORIZED 0
        )CODE");
    }
}


void CodegenNeuronCppVisitor::print_mechanism_variables_macros() {
    printer->add_newline();
    printer->add_line("static constexpr auto number_of_datum_variables = ",
                      std::to_string(int_variables_size()),
                      ";");
    printer->add_line("static constexpr auto number_of_floating_point_variables = ",
                      std::to_string(float_variables_size()),
                      ";");
    printer->add_newline();
    printer->add_multi_line(R"CODE(
    namespace {
    template <typename T>
    using _nrn_mechanism_std_vector = std::vector<T>;
    using _nrn_model_sorted_token = neuron::model_sorted_token;
    using _nrn_mechanism_cache_range =
        neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
    using _nrn_mechanism_cache_instance =
        neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
    template <typename T>
    using _nrn_mechanism_field = neuron::mechanism::field<T>;
    template <typename... Args>
    void _nrn_mechanism_register_data_fields(Args&&... args) {
        neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
    }
    }  // namespace
    )CODE");
    printer->add_line("/* NEURON RANGE variables macro definitions */");
    for (auto i = 0; i < codegen_float_variables.size(); ++i) {
        const auto float_var = codegen_float_variables[i];
        if (float_var->is_array()) {
            printer->add_line("#define ",
                              float_var->get_name(),
                              "(id) _ml->template data_array<",
                              std::to_string(i),
                              ", ",
                              std::to_string(float_var->get_length()),
                              ">(id)");
        } else {
            printer->add_line("#define ",
                              float_var->get_name(),
                              "(id) _ml->template fpfield<",
                              std::to_string(i),
                              ">(id)");
        }
    }
    printer->add_line("/* NEURON GLOBAL variables macro definitions */");
    // Go through the area (if point_process), ions
    // TODO: More prints here?
}


void CodegenNeuronCppVisitor::print_mechanism_global_var_structure(bool print_initializers) {
    // TODO: Print only global variables printed in NEURON
    printer->add_line();
    printer->add_line("/* NEURON global variables */");
    if (info.primes_size != 0) {
        printer->fmt_line("static neuron::container::field_index _slist1[{0}], _dlist1[{0}];",
                          info.primes_size);
    }
}


void CodegenNeuronCppVisitor::print_prcellstate_macros() const {
    printer->add_line("#ifndef NRN_PRCELLSTATE");
    printer->add_line("#define NRN_PRCELLSTATE 0");
    printer->add_line("#endif");
}


void CodegenNeuronCppVisitor::print_mechanism_info() {
    auto variable_printer = [&](const std::vector<SymbolType>& variables) {
        for (const auto& v: variables) {
            auto name = v->get_name();
            if (!info.point_process) {
                name += "_" + info.mod_suffix;
            }
            if (v->is_array()) {
                name += fmt::format("[{}]", v->get_length());
            }
            printer->add_line(add_escape_quote(name), ",");
        }
    };

    printer->add_newline(2);
    printer->add_line("/** channel information */");
    printer->add_line("static const char *mechanism[] = {");
    printer->increase_indent();
    printer->add_line(add_escape_quote(nmodl_version()), ",");
    printer->add_line(add_escape_quote(info.mod_suffix), ",");
    variable_printer(info.range_parameter_vars);
    printer->add_line("0,");
    variable_printer(info.range_assigned_vars);
    printer->add_line("0,");
    variable_printer(info.range_state_vars);
    printer->add_line("0,");
    variable_printer(info.pointer_variables);
    printer->add_line("0");
    printer->decrease_indent();
    printer->add_line("};");
}


void CodegenNeuronCppVisitor::print_global_variables_for_hoc() {
    // TODO: Write HocParmLimits and other HOC global variables (delta_t)
}


int CodegenNeuronCppVisitor::position_of_float_var(const std::string& name) const {
    const auto has_name = [&name](const SymbolType& symbol) { return symbol->get_name() == name; };
    const auto var_iter =
        std::find_if(codegen_float_variables.begin(), codegen_float_variables.end(), has_name);
    if (var_iter != codegen_float_variables.end()) {
        return var_iter - codegen_float_variables.begin();
    } else {
        throw std::logic_error(name + " variable not found");
    }
}


int CodegenNeuronCppVisitor::position_of_int_var(const std::string& name) const {
    const auto has_name = [&name](const IndexVariableInfo& index_var_symbol) {
        return index_var_symbol.symbol->get_name() == name;
    };
    const auto var_iter =
        std::find_if(codegen_int_variables.begin(), codegen_int_variables.end(), has_name);
    if (var_iter != codegen_int_variables.end()) {
        return var_iter - codegen_int_variables.begin();
    } else {
        throw std::logic_error(name + " variable not found");
    }
}


void CodegenNeuronCppVisitor::print_sdlists_init(bool print_initializers) {
    for (auto i = 0; i < info.prime_variables_by_order.size(); ++i) {
        const auto& prime_var = info.prime_variables_by_order[i];
        // TODO: Something similar needs to happen for slist/dlist2 but I don't know their usage at
        // the moment
        // TODO: We have to do checks and add errors similar to nocmodl in the
        // SemanticAnalysisVisitor
        if (prime_var->is_array()) {
            // TODO: Needs a for loop here. Look at
            // https://github.com/neuronsimulator/nrn/blob/df001a436bcb4e23d698afe66c2a513819a6bfe8/src/nmodl/deriv.cpp#L524
            // TODO: Also needs a test
            printer->fmt_push_block("for (int _i = 0; _i < {}; ++_i)", prime_var->get_length());
            printer->fmt_line("/* {}[{}] */", prime_var->get_name(), prime_var->get_length());
            printer->fmt_line("_slist1[{}+_i] = {{{}, _i}}",
                              i,
                              position_of_float_var(prime_var->get_name()));
            const auto prime_var_deriv_name = "D" + prime_var->get_name();
            printer->fmt_line("/* {}[{}] */", prime_var_deriv_name, prime_var->get_length());
            printer->fmt_line("_dlist1[{}+_i] = {{{}, _i}}",
                              i,
                              position_of_float_var(prime_var_deriv_name));
            printer->pop_block();
        } else {
            printer->fmt_line("/* {} */", prime_var->get_name());
            printer->fmt_line("_slist1[{}] = {{{}, 0}}",
                              i,
                              position_of_float_var(prime_var->get_name()));
            const auto prime_var_deriv_name = "D" + prime_var->get_name();
            printer->fmt_line("/* {} */", prime_var_deriv_name);
            printer->fmt_line("_dlist1[{}] = {{{}, 0}}",
                              i,
                              position_of_float_var(prime_var_deriv_name));
        }
    }
}


void CodegenNeuronCppVisitor::print_mechanism_register() {
    // TODO: Write this according to NEURON
    printer->add_newline(2);
    printer->add_line("/** register channel with the simulator */");
    printer->fmt_push_block("void _{}_reg()", info.mod_file);
    print_sdlists_init(true);
    // type related information
    auto suffix = add_escape_quote(info.mod_suffix);
    printer->add_newline();
    printer->fmt_line("int mech_type = nrn_get_mechtype({});", suffix);

    // More things to add here
    printer->add_line("_nrn_mechanism_register_data_fields(_mechtype,");
    printer->increase_indent();
    const auto codegen_float_variables_size = codegen_float_variables.size();
    for (int i = 0; i < codegen_float_variables_size; ++i) {
        const auto& float_var = codegen_float_variables[i];
        const auto print_comma = i < codegen_float_variables_size - 1 || info.emit_cvode;
        if (float_var->is_array()) {
            printer->fmt_line("_nrn_mechanism_field<double>{{\"{}\", {}}} /* {} */{}",
                              float_var->get_name(),
                              float_var->get_length(),
                              i,
                              print_comma ? "," : "");
        } else {
            printer->fmt_line("_nrn_mechanism_field<double>{{\"{}\"}} /* {} */{}",
                              float_var->get_name(),
                              i,
                              print_comma ? "," : "");
        }
    }
    if (info.emit_cvode) {
        printer->add_line("_nrn_mechanism_field<int>{\"_cvode_ieq\", \"cvodeieq\"} /* 0 */");
    }
    printer->decrease_indent();
    printer->add_line(");");
    printer->add_newline();
    printer->pop_block();
}


void CodegenNeuronCppVisitor::print_mechanism_range_var_structure(bool print_initializers) {
    // TODO: Print macros
}


// TODO: Needs changes
void CodegenNeuronCppVisitor::print_global_function_common_code(BlockType type,
                                                                const std::string& function_name) {
    return;
}


void CodegenNeuronCppVisitor::print_nrn_constructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Constructor);
    if (info.constructor_node != nullptr) {
        const auto& block = info.constructor_node->get_statement_block();
        print_statement_block(*block, false, false);
    }
    printer->add_line("#endif");
    // printer->pop_block();
}


void CodegenNeuronCppVisitor::print_nrn_destructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Destructor);
    if (info.destructor_node != nullptr) {
        const auto& block = info.destructor_node->get_statement_block();
        print_statement_block(*block, false, false);
    }
    printer->add_line("#endif");
    // printer->pop_block();
}


// TODO: Print the equivalent of `nrn_alloc_<mech_name>`
void CodegenNeuronCppVisitor::print_nrn_alloc() {
    printer->add_newline(2);
    auto method = method_name(naming::NRN_ALLOC_METHOD);
    printer->fmt_push_block("static void {}(double* data, Datum* indexes, int type)", method);
    printer->add_line("// do nothing");
    printer->pop_block();
}


void CodegenNeuronCppVisitor::visit_solution_expression(const SolutionExpression& node) {
    auto block = node.get_node_to_solve().get();
    if (block->is_statement_block()) {
        auto statement_block = dynamic_cast<ast::StatementBlock*>(block);
        print_statement_block(*statement_block, false, false);
    } else {
        block->accept(*this);
    }
}


/****************************************************************************************/
/*                                 Print nrn_state routine                              */
/****************************************************************************************/


void CodegenNeuronCppVisitor::print_nrn_state() {
    if (!nrn_state_required()) {
        return;
    }
    codegen = true;

    printer->add_line("nrn_state");
    // TODO: Write for NEURON

    codegen = false;
}


/****************************************************************************************/
/*                              Print nrn_cur related routines                          */
/****************************************************************************************/


void CodegenNeuronCppVisitor::print_nrn_current(const BreakpointBlock& node) {
    return;
}


void CodegenNeuronCppVisitor::print_nrn_cur_conductance_kernel(const BreakpointBlock& node) {
    return;
}


void CodegenNeuronCppVisitor::print_nrn_cur_non_conductance_kernel() {
    return;
}


void CodegenNeuronCppVisitor::print_nrn_cur_kernel(const BreakpointBlock& node) {
    return;
}


void CodegenNeuronCppVisitor::print_fast_imem_calculation() {
    return;
}


void CodegenNeuronCppVisitor::print_nrn_cur() {
    if (!nrn_cur_required()) {
        return;
    }

    codegen = true;

    codegen = false;
}


/****************************************************************************************/
/*                            Main code printing entry points                            */
/****************************************************************************************/

void CodegenNeuronCppVisitor::print_headers_include() {
    print_standard_includes();
    print_neuron_includes();
}


void CodegenNeuronCppVisitor::print_macro_definitions() {
    print_global_macros();
    print_mechanism_variables_macros();
}


void CodegenNeuronCppVisitor::print_namespace_begin() {
    print_namespace_start();
}


void CodegenNeuronCppVisitor::print_namespace_end() {
    print_namespace_stop();
}


void CodegenNeuronCppVisitor::print_data_structures(bool print_initializers) {
    print_mechanism_global_var_structure(print_initializers);
    print_mechanism_range_var_structure(print_initializers);
}


void CodegenNeuronCppVisitor::print_v_unused() const {
    if (!info.vectorize) {
        return;
    }
    printer->add_multi_line(R"CODE(
        #if NRN_PRCELLSTATE
        inst->v_unused[id] = v;
        #endif
    )CODE");
}


void CodegenNeuronCppVisitor::print_g_unused() const {
    printer->add_multi_line(R"CODE(
        #if NRN_PRCELLSTATE
        inst->g_unused[id] = g;
        #endif
    )CODE");
}

// TODO: Print functions, procedues and nrn_state
void CodegenNeuronCppVisitor::print_compute_functions() {
    // for (const auto& procedure: info.procedures) {
    //     print_procedure(*procedure); // maybes yes
    // }
    // for (const auto& function: info.functions) {
    //     print_function(*function); // maybe yes
    // }
    print_nrn_state();  // Only this
}


void CodegenNeuronCppVisitor::print_codegen_routines() {
    codegen = true;
    print_backend_info();
    print_headers_include();
    print_macro_definitions();
    print_namespace_begin();
    print_nmodl_constants();
    print_prcellstate_macros();
    print_mechanism_info();       // same as NEURON
    print_data_structures(true);  // print macros instead here for range variables and global ones
    print_global_variables_for_hoc();   // same
    print_memory_allocation_routine();  // same
    print_abort_routine();              // simple
    print_nrn_alloc();                  // `nrn_alloc_hh`
    // print_nrn_constructor(); // should be same
    // print_nrn_destructor(); // should be same
    print_function_prototypes();  // yes
    print_compute_functions();    // only functions, procedures and state
    print_mechanism_register();   // Yes
    print_namespace_end();        // Yes
    codegen = false;
}

}  // namespace codegen
}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/codegen_info.hpp"

#include "ast/all.hpp"
#include "utils/logger.hpp"
#include "visitors/var_usage_visitor.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {
namespace codegen {

using symtab::syminfo::NmodlType;
using visitor::VarUsageVisitor;

SymbolType make_symbol(const std::string& name) {
    return std::make_shared<symtab::Symbol>(name, ModToken());
}


std::string shadow_varname(const std::string& name) {
    return "shadow_" + name;
}


/// if any ion has write variable
bool CodegenInfo::ion_has_write_variable() const {
    for (const auto& ion: ions) {
        if (!ion.writes.empty()) {
            return true;
        }
    }
    return false;
}


/// if given variable is ion write variable
bool CodegenInfo::is_ion_write_variable(const std::string& name) const {
    for (const auto& ion: ions) {
        for (auto& var: ion.writes) {
            if (var == name) {
                return true;
            }
        }
    }
    return false;
}


/// if given variable is ion read variable
bool CodegenInfo::is_ion_read_variable(const std::string& name) const {
    for (const auto& ion: ions) {
        for (auto& var: ion.reads) {
            if (var == name) {
                return true;
            }
        }
    }
    return false;
}


/// if either read or write variable
bool CodegenInfo::is_ion_variable(const std::string& name) const {
    return is_ion_read_variable(name) || is_ion_write_variable(name);
}


/// if a current (ionic or non-specific)
bool CodegenInfo::is_current(const std::string& name) const {
    for (auto& var: currents) {
        if (var == name) {
            return true;
        }
    }
    return false;
}

/// true is a given variable name if a ionic current
/// (i.e. currents excluding non-specific current)
bool CodegenInfo::is_ionic_current(const std::string& name) const {
    for (const auto& ion: ions) {
        if (ion.is_ionic_current(name) == true) {
            return true;
        }
    }
    return false;
}

/// true if given variable name is a ionic concentration
bool CodegenInfo::is_ionic_conc(const std::string& name) const {
    for (const auto& ion: ions) {
        if (ion.is_ionic_conc(name) == true) {
            return true;
        }
    }
    return false;
}

bool CodegenInfo::function_uses_table(std::string& name) const {
    for (auto& function: functions_with_table) {
        if (name == function->get_node_name()) {
            return true;
        }
    }
    return false;
}

/**
 * Check if NrnState node in the AST has EigenSolverBlock node
 *
 * @return True if EigenSolverBlock exist in the node
 */
bool CodegenInfo::nrn_state_has_eigen_solver_block() const {
    if (nrn_state_block == nullptr) {
        return false;
    }
    return !collect_nodes(*nrn_state_block, {ast::AstNodeType::EIGEN_NEWTON_SOLVER_BLOCK}).empty();
}

/**
 * Check if WatchStatement uses voltage variable v
 *
 * Watch statement has condition expression which could use voltage
 * variable `v`. To avoid memory access into voltage array we check
 * if `v` is used and then print necessary code.
 *
 * @return true if voltage variable b is used otherwise false
 */
bool CodegenInfo::is_voltage_used_by_watch_statements() const {
    for (const auto& statement: watch_statements) {
        auto v_used = VarUsageVisitor().variable_used(*statement, "v");
        if (v_used) {
            return true;
        }
    }
    return false;
}

bool CodegenInfo::state_variable(const std::string& name) const {
    // clang-format off
    auto result = std::find_if(state_vars.begin(),
                               state_vars.end(),
                               [&name](const SymbolType& sym) {
                                   return name == sym->get_name();
                               }
    );
    // clang-format on
    return result != state_vars.end();
}

std::pair<std::string, std::string> CodegenInfo::read_ion_variable_name(
    const std::string& name) const {
    return {name, "ion_" + name};
}


std::pair<std::string, std::string> CodegenInfo::write_ion_variable_name(
    const std::string& name) const {
    return {"ion_" + name, name};
}


/**
 * \details Current variable used in breakpoint block could be local variable.
 * In this case, neuron has already renamed the variable name by prepending
 * "_l". In our implementation, the variable could have been renamed by
 * one of the pass. And hence, we search all local variables and check if
 * the variable is renamed. Note that we have to look into the symbol table
 * of statement block and not breakpoint.
 */
std::string CodegenInfo::breakpoint_current(std::string current) const {
    auto& breakpoint = breakpoint_node;
    if (breakpoint == nullptr) {
        return current;
    }
    const auto& symtab = breakpoint->get_statement_block()->get_symbol_table();
    const auto& variables = symtab->get_variables_with_properties(NmodlType::local_var);
    for (const auto& var: variables) {
        std::string renamed_name = var->get_name();
        std::string original_name = var->get_original_name();
        if (current == original_name) {
            current = renamed_name;
            break;
        }
    }
    return current;
}


bool CodegenInfo::is_an_instance_variable(const std::string& varname) const {
    /// check if symbol of given name exist
    auto check_symbol = [](const std::string& name, const std::vector<SymbolType>& symbols) {
        for (auto& symbol: symbols) {
            if (symbol->get_name() == name) {
                return true;
            }
        }
        return false;
    };

    /// check if variable exist into all possible types
    if (check_symbol(varname, assigned_vars) || check_symbol(varname, state_vars) ||
        check_symbol(varname, range_parameter_vars) || check_symbol(varname, range_assigned_vars) ||
        check_symbol(varname, range_state_vars)) {
        return true;
    }
    return false;
}


/**
 * IndexVariableInfo has following constructor arguments:
 *      - symbol
 *      - is_vdata   (false)
 *      - is_index   (false
 *      - is_integer (false)
 *
 * Which variables are constant qualified?
 *
 *  - node area is read only
 *  - read ion variables are read only
 *  - style_ionname is index / offset
 */
void CodegenInfo::get_int_variables() {
    if (point_process) {
        codegen_int_variables.emplace_back(make_symbol(naming::NODE_AREA_VARIABLE));
        codegen_int_variables.back().is_constant = true;
        /// note that this variable is not printed in neuron implementation
        if (artificial_cell) {
            codegen_int_variables.emplace_back(make_symbol(naming::POINT_PROCESS_VARIABLE), true);
        } else {
            codegen_int_variables.emplace_back(make_symbol(naming::POINT_PROCESS_VARIABLE),
                                               false,
                                               false,
                                               true);
            codegen_int_variables.back().is_constant = true;
        }
    }

    for (const auto& ion: ions) {
        bool need_style = false;
        std::unordered_map<std::string, int> ion_vars;  // used to keep track of the variables to
                                                        // not have doubles between read/write. Same
                                                        // name variables are allowed
        for (const auto& var: ion.reads) {
            const std::string name = naming::ION_VARNAME_PREFIX + var;
            codegen_int_variables.emplace_back(make_symbol(name));
            codegen_int_variables.back().is_constant = true;
            ion_vars[name] = codegen_int_variables.size() - 1;
        }

        /// symbol for di_ion_dv var
        std::shared_ptr<symtab::Symbol> ion_di_dv_var = nullptr;

        for (const auto& var: ion.writes) {
            const std::string name = naming::ION_VARNAME_PREFIX + var;

            const auto ion_vars_it = ion_vars.find(name);
            if (ion_vars_it != ion_vars.end()) {
                codegen_int_variables[ion_vars_it->second].is_constant = false;
            } else {
                codegen_int_variables.emplace_back(make_symbol(naming::ION_VARNAME_PREFIX + var));
            }
            if (ion.is_ionic_current(var)) {
                ion_di_dv_var = make_symbol(std::string(naming::ION_VARNAME_PREFIX) + "di" +
                                            ion.name + "dv");
            }
            if (ion.is_intra_cell_conc(var) || ion.is_extra_cell_conc(var)) {
                need_style = true;
            }
        }

        /// insert after read/write variables but before style ion variable
        if (ion_di_dv_var != nullptr) {
            codegen_int_variables.emplace_back(ion_di_dv_var);
        }

        if (need_style) {
            codegen_int_variables.emplace_back(make_symbol("style_" + ion.name), false, true);
            codegen_int_variables.back().is_constant = true;
        }
    }

    for (const auto& var: pointer_variables) {
        auto name = var->get_name();
        if (var->has_any_property(NmodlType::pointer_var)) {
            codegen_int_variables.emplace_back(make_symbol(name));
        } else {
            codegen_int_variables.emplace_back(make_symbol(name), true);
        }
    }

    if (diam_used) {
        codegen_int_variables.emplace_back(make_symbol(naming::DIAM_VARIABLE));
    }

    if (area_used) {
        codegen_int_variables.emplace_back(make_symbol(naming::AREA_VARIABLE));
    }

    // for non-artificial cell, when net_receive buffering is enabled
    // then tqitem is an offset
    if (net_send_used) {
        if (artificial_cell) {
            codegen_int_variables.emplace_back(make_symbol(naming::TQITEM_VARIABLE), true);
        } else {
            codegen_int_variables.emplace_back(make_symbol(naming::TQITEM_VARIABLE),
                                               false,
                                               false,
                                               true);
            codegen_int_variables.back().is_constant = true;
        }
        tqitem_index = codegen_int_variables.size() - 1;
    }

    /**
     * \note Variables for watch statements : there is one extra variable
     * used in coreneuron compared to actual watch statements for compatibility
     * with neuron (which uses one extra Datum variable)
     */
    if (!watch_statements.empty()) {
        for (int i = 0; i < watch_statements.size() + 1; i++) {
            codegen_int_variables.emplace_back(make_symbol(fmt::format("watch{}", i)),
                                               false,
                                               false,
                                               true);
        }
    }
}


/**
 * \details When we enable fine level parallelism at channel level, we have do updates
 * to ion variables in atomic way. As cpus don't have atomic instructions in
 * simd loop, we have to use shadow vectors for every ion variables. Here
 * we return list of all such variables.
 *
 * \todo If conductances are specified, we don't need all below variables
 */
void CodegenInfo::get_shadow_variables() {
    for (const auto& ion: ions) {
        for (const auto& var: ion.writes) {
            codegen_shadow_variables.push_back(
                {make_symbol(shadow_varname(naming::ION_VARNAME_PREFIX + var))});
            if (ion.is_ionic_current(var)) {
                codegen_shadow_variables.push_back({make_symbol(shadow_varname(
                    std::string(naming::ION_VARNAME_PREFIX) + "di" + ion.name + "dv"))});
            }
        }
    }
    codegen_shadow_variables.push_back({make_symbol("ml_rhs")});
    codegen_shadow_variables.push_back({make_symbol("ml_d")});
}


void CodegenInfo::get_float_variables() {
    // sort with definition order
    auto comparator = [](const SymbolType& first, const SymbolType& second) -> bool {
        return first->get_definition_order() < second->get_definition_order();
    };

    auto assigned = assigned_vars;
    auto states = state_vars;

    // each state variable has corresponding Dstate variable
    for (auto& state: states) {
        auto name = "D" + state->get_name();
        auto symbol = make_symbol(name);
        if (state->is_array()) {
            symbol->set_as_array(state->get_length());
        }
        symbol->set_definition_order(state->get_definition_order());
        assigned.push_back(symbol);
    }
    std::sort(assigned.begin(), assigned.end(), comparator);

    codegen_float_variables = range_parameter_vars;
    codegen_float_variables.insert(codegen_float_variables.end(),
                                   range_assigned_vars.begin(),
                                   range_assigned_vars.end());
    codegen_float_variables.insert(codegen_float_variables.end(),
                                   range_state_vars.begin(),
                                   range_state_vars.end());
    codegen_float_variables.insert(codegen_float_variables.end(), assigned.begin(), assigned.end());

    if (vectorize) {
        codegen_float_variables.push_back(make_symbol(naming::VOLTAGE_UNUSED_VARIABLE));
    }
    if (breakpoint_exist()) {
        std::string name = vectorize ? naming::CONDUCTANCE_UNUSED_VARIABLE
                                     : naming::CONDUCTANCE_VARIABLE;
        codegen_float_variables.push_back(make_symbol(name));
    }
    if (net_receive_exist()) {
        codegen_float_variables.push_back(make_symbol(naming::T_SAVE_VARIABLE));
    }
}

/**
 * \details Certain statements like unit, comment, solve can/need to be skipped
 * during code generation. Note that solve block is wrapped in expression
 * statement and hence we have to check inner expression. It's also true
 * for the initial block defined inside net receive block.
 */
bool CodegenInfo::statement_to_skip(const ast::Statement& node) const {
    // clang-format off
    if (node.is_unit_state()
        || node.is_line_comment()
        || node.is_block_comment()
        || node.is_solve_block()
        || node.is_conductance_hint()
        || node.is_table_statement()) {
        return true;
    }
    // clang-format on
    if (node.is_expression_statement()) {
        auto expression = dynamic_cast<const ast::ExpressionStatement*>(&node)->get_expression();
        if (expression->is_solve_block()) {
            return true;
        }
        if (expression->is_initial_block()) {
            return true;
        }
    }
    return false;
}

}  // namespace codegen
}  // namespace nmodl

/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/codegen_info.hpp"

#include "ast/all.hpp"
#include "visitors/var_usage_visitor.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {
namespace codegen {

using namespace fmt::literals;
using visitor::VarUsageVisitor;
using symtab::syminfo::NmodlType;

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
 * \details Depending upon the block type, we have to print read/write ion variables
 * during code generation. Depending on block/procedure being printed, this
 * method return statements as vector. As different code backends could have
 * different variable names, we rely on backend-specific read_ion_variable_name
 * and write_ion_variable_name method which will be overloaded.
 *
 * \todo After looking into mod2c and neuron implementation, it seems like
 * Ode block type is not used (?). Need to look into implementation details.
 */
std::vector<std::string> CodegenInfo::ion_read_statements(BlockType type) {
    //if (optimize_ion_variable_copies()) {
    //    return ion_read_statements_optimized(type);
    //}
    std::vector<std::string> statements;
    for (const auto& ion: ions) {
        auto name = ion.name;
        for (const auto& var: ion.reads) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && state_variable(var)) {
                continue;
            }
            auto variable_names = read_ion_variable_name(var);
            auto first = variable_names.first;
            auto second = variable_names.second;
            statements.push_back("{} = {}"_format(first, second));
        }
        for (const auto& var: ion.writes) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && state_variable(var)) {
                continue;
            }
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = variables.first;
                auto second = variables.second;
                statements.push_back("{} = {}"_format(first, second));
            }
        }
    }
    return statements;
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
    auto breakpoint = breakpoint_node;
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

std::vector<ShadowUseStatement> CodegenInfo::ion_write_statements(BlockType type) {
    std::vector<ShadowUseStatement> statements;
    for (const auto& ion: ions) {
        std::string concentration;
        auto name = ion.name;
        for (const auto& var: ion.writes) {
            auto variable_names = write_ion_variable_name(var);
            if (ion.is_ionic_current(var)) {
                if (type == BlockType::Equation) {
                    auto current = breakpoint_current(var);
                    auto lhs = variable_names.first;
                    auto op = "+=";
                    auto rhs = current;
                    if (point_process) {
                        auto area = codegen::naming::NODE_AREA_VARIABLE;
                        rhs += "*(1.e2/{})"_format(area);
                    }
                    statements.push_back(ShadowUseStatement{lhs, op, rhs});
                }
            } else {
                if (!ion.is_rev_potential(var)) {
                    concentration = var;
                }
                auto lhs = variable_names.first;
                auto op = "=";
                auto rhs = variable_names.second;
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
                throw std::logic_error("codegen error for {} ion"_format(ion.name));
            }
            auto ion_type_name = "{}_type"_format(ion.name);
            auto lhs = "int {}"_format(ion_type_name);
            auto op = "=";
            auto rhs = ion_type_name;
            statements.push_back(ShadowUseStatement{lhs, op, rhs});
            // TODO
            //auto statement = conc_write_statement(ion.name, concentration, index);
            //statements.push_back(ShadowUseStatement{statement, "", ""});
        }
    }
    return statements;
}

}  // namespace codegen
}  // namespace nmodl

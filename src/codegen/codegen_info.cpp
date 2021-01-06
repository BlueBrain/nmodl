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

using namespace fmt::literals;
using symtab::syminfo::NmodlType;
using visitor::VarUsageVisitor;

/// if any ion has write variable
bool CodegenInfo::ion_has_write_variable() const {
    return std::any_of(ions.begin(), ions.end(), [](auto const& ion) {
        return !ion.writes.empty();
    });
}


/// if given variable is ion write variable
bool CodegenInfo::is_ion_write_variable(const std::string& name) const {
    return std::any_of(ions.begin(), ions.end(), [&name](auto const& ion) {
        return std::any_of(ion.writes.begin(), ion.writes.end(), [&name](auto const& var) {
            return var == name;
        });
    });
}


/// if given variable is ion read variable
bool CodegenInfo::is_ion_read_variable(const std::string& name) const {
    return std::any_of(ions.begin(), ions.end(), [&name](auto const& ion) {
        return std::any_of(ion.reads.begin(), ion.reads.end(), [&name](auto const& var) {
            return var == name;
        });
    });
}


/// if either read or write variable
bool CodegenInfo::is_ion_variable(const std::string& name) const {
    return is_ion_read_variable(name) || is_ion_write_variable(name);
}


/// if a current (ionic or non-specific)
bool CodegenInfo::is_current(const std::string& name) const {
    return std::any_of(currents.begin(), currents.end(), [&name](auto const& var) {
        return var == name;
    });
}

/// true is a given variable name if a ionic current
/// (i.e. currents excluding non-specific current)
bool CodegenInfo::is_ionic_current(const std::string& name) const {
    return std::any_of(ions.begin(), ions.end(), [&name](auto const& ion) {
        return ion.is_ionic_current(name);
    });
}

/// true if given variable name is a ionic concentration
bool CodegenInfo::is_ionic_conc(const std::string& name) const {
    return std::any_of(ions.begin(), ions.end(), [&name](auto const& ion) {
        return ion.is_ionic_conc(name);
    });
}

bool CodegenInfo::function_uses_table(std::string& name) const {
    return std::any_of(functions_with_table.begin(),
                       functions_with_table.end(),
                       [&name](auto const& function) { return name == function->get_node_name(); });
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
    return std::any_of(watch_statements.begin(), watch_statements.end(), [](auto const& statement) {
        return VarUsageVisitor{}.variable_used(*statement, "v");
    });
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

}  // namespace codegen
}  // namespace nmodl

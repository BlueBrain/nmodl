/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file token_mapping.hpp
 * \brief Map different tokens from lexer to token types
 */

#include <string>

#include "parser/nmodl/nmodl_parser.hpp"

namespace nmodl {

bool is_keyword(const std::string& name);
bool is_method(const std::string& name);

parser::NmodlParser::token_type token_type(const std::string& name);
std::vector<std::string> get_external_variables();
std::vector<std::string> get_external_functions();

namespace details {

/**
 * Definition type similar to old implementation
 *
 * In the original implementation of NMODL (mod2c, nocmodl) different vectors were
 * created for \c extdef, \c extdef2, \c extdef3, \c extdef4 etc. We are changing
 * those vectors with <c><name, type></c> map. This will help us to search
 * in single map and find it's type. The types are defined as follows:
 *
 * - DefinitionType::EXT_DOUBLE : external names that can be used as doubles
 *                                without giving an error message
 * - DefinitionType::EXT_2      : external function names that can be used with
 *                                array and function name arguments
 * - DefinitionType::EXT_3      : function names that get two reset arguments
 * - DefinitionType::EXT_4      : functions that need a first arg of \c NrnThread*
 * - DefinitionType::EXT_DOUBLE_4
 *                              : functions that need a first arg of \c NrnThread* and can be
 *                                used as doubles
 * - DefinitionType::EXT_5      : external definition names that are not \c threadsafe
 *
 */

enum class DefinitionType { EXT_DOUBLE, EXT_2, EXT_3, EXT_4, EXT_DOUBLE_4, EXT_5 };

extern const std::map<std::string, DefinitionType> extern_definitions;

/**
 * Checks if \c token is one of the functions coming from NEURON/CoreNEURON and needs
 * passing NrnThread* as first argument (typical name of variable \c nt)
 *
 * @param token Name of function
 * @return True or false depending if the function needs NrnThread* argument
 */
inline bool needs_neuron_thread_first_arg(const std::string& token) {
    auto extern_def = extern_definitions.find(token);
    return extern_def != extern_definitions.end() &&
           (extern_def->second == DefinitionType::EXT_4 ||
            extern_def->second == DefinitionType::EXT_DOUBLE_4);
}

}  // namespace details

}  // namespace nmodl

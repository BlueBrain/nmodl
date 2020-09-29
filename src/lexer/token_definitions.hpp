/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file token_definitions.hpp
 * \brief Map different definitions of tokens according to mod2c
 */

#include <map>

namespace nmodl {

/// details of lexer tokens
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
bool needs_neuron_thread_first_arg(const std::string& token);

}  // namespace details

}  // namespace nmodl
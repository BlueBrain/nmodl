/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/
#include <map>
#include <string>

#include "lexer/token_definitions.hpp"

namespace nmodl {

namespace details {

const std::map<std::string, DefinitionType> extern_definitions = {
    {"first_time", DefinitionType::EXT_DOUBLE},
    {"error", DefinitionType::EXT_DOUBLE},
    {"f_flux", DefinitionType::EXT_DOUBLE},
    {"b_flux", DefinitionType::EXT_DOUBLE},
    {"fabs", DefinitionType::EXT_DOUBLE},
    {"sqrt", DefinitionType::EXT_DOUBLE},
    {"sin", DefinitionType::EXT_DOUBLE},
    {"cos", DefinitionType::EXT_DOUBLE},
    {"tan", DefinitionType::EXT_DOUBLE},
    {"acos", DefinitionType::EXT_DOUBLE},
    {"asin", DefinitionType::EXT_DOUBLE},
    {"atan", DefinitionType::EXT_DOUBLE},
    {"atan2", DefinitionType::EXT_DOUBLE},
    {"sinh", DefinitionType::EXT_DOUBLE},
    {"cosh", DefinitionType::EXT_DOUBLE},
    {"tanh", DefinitionType::EXT_DOUBLE},
    {"floor", DefinitionType::EXT_DOUBLE},
    {"ceil", DefinitionType::EXT_DOUBLE},
    {"fmod", DefinitionType::EXT_DOUBLE},
    {"log10", DefinitionType::EXT_DOUBLE},
    {"log", DefinitionType::EXT_DOUBLE},
    {"pow", DefinitionType::EXT_DOUBLE},
    {"printf", DefinitionType::EXT_DOUBLE},
    {"prterr", DefinitionType::EXT_DOUBLE},
    {"exp", DefinitionType::EXT_DOUBLE},
    {"threshold", DefinitionType::EXT_DOUBLE},
    {"force", DefinitionType::EXT_DOUBLE},
    {"deflate", DefinitionType::EXT_DOUBLE},
    {"expfit", DefinitionType::EXT_DOUBLE},
    {"derivs", DefinitionType::EXT_DOUBLE},
    {"spline", DefinitionType::EXT_DOUBLE},
    {"hyperbol", DefinitionType::EXT_DOUBLE},
    {"revhyperbol", DefinitionType::EXT_DOUBLE},
    {"sigmoid", DefinitionType::EXT_DOUBLE},
    {"revsigmoid", DefinitionType::EXT_DOUBLE},
    {"harmonic", DefinitionType::EXT_DOUBLE},
    {"squarewave", DefinitionType::EXT_DOUBLE},
    {"sawtooth", DefinitionType::EXT_DOUBLE},
    {"revsawtooth", DefinitionType::EXT_DOUBLE},
    {"ramp", DefinitionType::EXT_DOUBLE},
    {"pulse", DefinitionType::EXT_DOUBLE},
    {"perpulse", DefinitionType::EXT_DOUBLE},
    {"step", DefinitionType::EXT_DOUBLE},
    {"perstep", DefinitionType::EXT_DOUBLE},
    {"erf", DefinitionType::EXT_DOUBLE},
    {"exprand", DefinitionType::EXT_DOUBLE},
    {"factorial", DefinitionType::EXT_DOUBLE},
    {"gauss", DefinitionType::EXT_DOUBLE},
    {"normrand", DefinitionType::EXT_DOUBLE},
    {"poisrand", DefinitionType::EXT_DOUBLE},
    {"poisson", DefinitionType::EXT_DOUBLE},
    {"setseed", DefinitionType::EXT_DOUBLE},
    {"scop_random", DefinitionType::EXT_DOUBLE},
    {"boundary", DefinitionType::EXT_DOUBLE},
    {"romberg", DefinitionType::EXT_DOUBLE},
    {"legendre", DefinitionType::EXT_DOUBLE},
    {"invert", DefinitionType::EXT_DOUBLE},
    {"stepforce", DefinitionType::EXT_DOUBLE},
    {"schedule", DefinitionType::EXT_DOUBLE},
    {"set_seed", DefinitionType::EXT_DOUBLE},
    {"nrn_pointing", DefinitionType::EXT_DOUBLE},
    {"state_discontinuity", DefinitionType::EXT_DOUBLE},
    {"net_send", DefinitionType::EXT_DOUBLE},
    {"net_move", DefinitionType::EXT_DOUBLE},
    {"net_event", DefinitionType::EXT_DOUBLE},
    {"nrn_random_play", DefinitionType::EXT_DOUBLE},
    {"nrn_ghk", DefinitionType::EXT_DOUBLE},
    {"romberg", DefinitionType::EXT_2},
    {"legendre", DefinitionType::EXT_2},
    {"deflate", DefinitionType::EXT_2},
    {"threshold", DefinitionType::EXT_3},
    {"squarewave", DefinitionType::EXT_3},
    {"sawtooth", DefinitionType::EXT_3},
    {"revsawtooth", DefinitionType::EXT_3},
    {"ramp", DefinitionType::EXT_3},
    {"pulse", DefinitionType::EXT_3},
    {"perpulse", DefinitionType::EXT_3},
    {"step", DefinitionType::EXT_3},
    {"perstep", DefinitionType::EXT_3},
    {"stepforce", DefinitionType::EXT_3},
    {"schedule", DefinitionType::EXT_3},
    {"at_time", DefinitionType::EXT_DOUBLE_4},
    {"force", DefinitionType::EXT_5},
    {"deflate", DefinitionType::EXT_5},
    {"expfit", DefinitionType::EXT_5},
    {"derivs", DefinitionType::EXT_5},
    {"spline", DefinitionType::EXT_5},
    {"exprand", DefinitionType::EXT_5},
    {"gauss", DefinitionType::EXT_5},
    {"normrand", DefinitionType::EXT_5},
    {"poisrand", DefinitionType::EXT_5},
    {"poisson", DefinitionType::EXT_5},
    {"setseed", DefinitionType::EXT_5},
    {"scop_random", DefinitionType::EXT_5},
    {"boundary", DefinitionType::EXT_5},
    {"romberg", DefinitionType::EXT_5},
    {"invert", DefinitionType::EXT_5},
    {"stepforce", DefinitionType::EXT_5},
    {"schedule", DefinitionType::EXT_5},
    {"set_seed", DefinitionType::EXT_5},
    {"nrn_random_play", DefinitionType::EXT_5}};

bool needs_neuron_thread_first_arg(const std::string& token) {
    auto extern_def = extern_definitions.find(token);
    if (extern_def != details::extern_definitions.end() &&
        (extern_def->second == details::DefinitionType::EXT_4 ||
         extern_def->second == details::DefinitionType::EXT_DOUBLE_4)) {
        return true;
    } else {
        return false;
    }
}

}  // namespace details

}  // namespace nmodl
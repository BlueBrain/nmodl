/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "symtab/symbol.hpp"
#include "fmt/format.h"


using namespace fmt::literals;

namespace nmodl {
namespace symtab {

using syminfo::NmodlType;
using syminfo::Status;


bool Symbol::is_variable() {
    // if symbol has one of the following property then it
    // is considered as variable in the NMODL
    // clang-format off
        NmodlType var_properties = NmodlType::local_var
                                    | NmodlType::global_var
                                    | NmodlType::range_var
                                    | NmodlType::param_assign
                                    | NmodlType::pointer_var
                                    | NmodlType::bbcore_pointer_var
                                    | NmodlType::extern_var
                                    | NmodlType::dependent_def
                                    | NmodlType::read_ion_var
                                    | NmodlType::write_ion_var
                                    | NmodlType::nonspecific_cur_var
                                    | NmodlType::electrode_cur_var
                                    | NmodlType::section_var
                                    | NmodlType::argument
                                    | NmodlType::extern_neuron_variable;
    // clang-format on
    return has_any_property(var_properties);
}

std::string Symbol::to_string() {
    std::string s(name);
    if (properties != NmodlType::empty) {
        s += " [Properties : {}]"_format(syminfo::to_string(properties));
    }
    if (status != Status::empty) {
        s += " [Status : {}]"_format(syminfo::to_string(status));
    }
    return s;
}

}  // namespace symtab
}  // namespace nmodl

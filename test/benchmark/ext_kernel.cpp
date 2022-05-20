/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "ext_kernel.hpp"

#include <iostream>

// external kernel stub
void nrn_state_hh_ext(void*) {
    throw std::runtime_error(
        "Error: this should have been external nrn_state_hh_ext kernel, check library and "
        "LD_LIBRARY_PATH\n");
}

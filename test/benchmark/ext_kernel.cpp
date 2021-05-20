/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "ext_kernel.hpp"

#include <iostream>

// external kernel stub
void nrn_state_hh_ext(void* ){
    std::cout << "stub kernel" << '\n';
}
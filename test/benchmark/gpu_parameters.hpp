/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief GPU execution parameters struct
 *
 * \file
 * \brief \copybrief nmodl::cuda_details::GPUExecutionParameters
 */

namespace nmodl {
namespace cuda_details {

struct GPUExecutionParameters {
    int gridDimX;
    int blockDimX;
};

}  // namespace cuda_details
}  // namespace nmodl

/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief Implement logger based on spdlog library
 */

#include <spdlog/spdlog.h>

namespace nmodl {

using logger_type = std::shared_ptr<spdlog::logger>;
extern logger_type logger;

}  // namespace nmodl

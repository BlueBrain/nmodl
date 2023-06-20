/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * \file
 * \brief Implement logger based on spdlog library
 */

// clang-format off
// disable clang-format to keep order of inclusion
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
// clang-format on

namespace nmodl {

using logger_type = std::shared_ptr<spdlog::logger>;
extern logger_type logger;

}  // namespace nmodl

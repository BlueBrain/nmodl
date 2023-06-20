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

#include <memory>

#include "utils/logger.hpp"

/**
 * \file
 * \brief \copybrief nmodl::Logger
 */

namespace nmodl {

using logger_type = std::shared_ptr<spdlog::logger>;

/**
 * \brief Logger implementation based on spdlog
 */
struct Logger {
    logger_type logger;
    Logger(const std::string& name, std::string pattern) {
        logger = spdlog::stdout_color_mt(name);
        logger->set_pattern(std::move(pattern));
    }
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
Logger nmodl_logger("NMODL", "[%n] [%^%l%$] :: %v");
logger_type logger = nmodl_logger.logger;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

}  // namespace nmodl

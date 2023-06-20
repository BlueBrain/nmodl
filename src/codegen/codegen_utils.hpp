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
 * \brief Implement utility functions for codegen visitors
 *
 */

#include <string>

namespace nmodl {
namespace codegen {
namespace utils {

/**
 * Handles the double constants format being printed in the generated code.
 *
 * It takes care of printing the values with the correct floating point precision
 * for each backend, similar to mod2c and Neuron.
 * This function can be called using as template `CodegenCVisitor`
 *
 * \param s_value The double constant as string
 * \return        The proper string to be printed in the generated file.
 */
template <typename T>
std::string format_double_string(const std::string& s_value);


/**
 * Handles the float constants format being printed in the generated code.
 *
 * It takes care of printing the values with the correct floating point precision
 * for each backend, similar to mod2c and Neuron.
 * This function can be called using as template `CodegenCVisitor`
 *
 * \param s_value The double constant as string
 * \return        The proper string to be printed in the generated file.
 */
template <typename T>
std::string format_float_string(const std::string& s_value);

}  // namespace utils
}  // namespace codegen
}  // namespace nmodl

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

#include "codegen/codegen_utils.hpp"

#include "codegen/codegen_cpp_visitor.hpp"

namespace nmodl {
namespace codegen {
namespace utils {
/**
 * \details We can directly print value but if user specify value as integer then
 * then it gets printed as an integer. To avoid this, we use below wrappers.
 * If user has provided integer then it gets printed as 1.0 (similar to mod2c
 * and neuron where ".0" is appended). Otherwise we print double variables as
 * they are represented in the mod file by user. If the value is in scientific
 * representation (1e+20, 1E-15) then keep it as it is.
 */
template <>
std::string format_double_string<CodegenCVisitor>(const std::string& s_value) {
    double value = std::stod(s_value);
    if (std::ceil(value) == value && s_value.find_first_of("eE") == std::string::npos) {
        return fmt::format("{:.1f}", value);
    }
    return s_value;
}


template <>
std::string format_float_string<CodegenCVisitor>(const std::string& s_value) {
    float value = std::stof(s_value);
    if (std::ceil(value) == value && s_value.find_first_of("eE") == std::string::npos) {
        return fmt::format("{:.1f}", value);
    }
    return s_value;
}
}  // namespace utils
}  // namespace codegen
}  // namespace nmodl

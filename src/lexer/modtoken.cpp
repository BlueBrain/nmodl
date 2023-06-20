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

#include "lexer/modtoken.hpp"

namespace nmodl {

using LocationType = nmodl::parser::location;

std::string ModToken::position() const {
    std::stringstream ss;
    if (external) {
        ss << "EXTERNAL";
    } else if (start_line() == 0) {
        ss << "UNKNOWN";
    } else {
        ss << pos;
    }
    return ss.str();
}

std::ostream& operator<<(std::ostream& stream, const ModToken& mt) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    stream << std::setw(15) << mt.name << " at [" << mt.position() << "]";
    return stream << " type " << mt.token;
}

ModToken operator+(ModToken const& adder1, ModToken const& adder2) {
    LocationType sum_pos = adder1.pos + adder2.pos;
    ModToken sum(adder1.name, adder1.token, sum_pos);

    return sum;
}

}  // namespace nmodl

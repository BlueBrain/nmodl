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

#include "printer/nmodl_printer.hpp"
#include "utils/string_utils.hpp"

namespace nmodl {
namespace printer {

NMODLPrinter::NMODLPrinter(const std::string& filename) {
    if (filename.empty()) {
        throw std::runtime_error("Empty filename for NMODLPrinter");
    }

    ofs.open(filename.c_str());

    if (ofs.fail()) {
        auto msg = "Error while opening file '" + filename + "' for NMODLPrinter";
        throw std::runtime_error(msg);
    }

    sbuf = ofs.rdbuf();
    result = std::make_shared<std::ostream>(sbuf);
}

void NMODLPrinter::push_level() {
    indent_level++;
    *result << "{";
    add_newline();
}

void NMODLPrinter::add_indent() {
    *result << std::string(indent_level * 4, ' ');
}

void NMODLPrinter::add_element(const std::string& name) {
    *result << name << "";
}

void NMODLPrinter::add_newline() {
    *result << std::endl;
}

void NMODLPrinter::pop_level() {
    indent_level--;
    add_indent();
    *result << "}";
}

}  // namespace printer
}  // namespace nmodl

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

#include "visitors/nmodl_visitor.hpp"

#include "visitors/visitor_utils.hpp"


namespace nmodl {
namespace visitor {

/** Helper function to visit vector elements
 *
 * @tparam T
 * @param elements vector of nodes/elements
 * @param separator separator to print for individual vector element
 * @param program  true if provided elements belong to program node
 * @param statement true if elements in vector of statement type
 */

template <typename T>
void NmodlPrintVisitor::visit_element(const std::vector<T>& elements,
                                      const std::string& separator,
                                      bool program,
                                      bool statement) {
    for (auto iter = elements.begin(); iter != elements.end(); iter++) {
        /// statements need indentation at the start
        if (statement) {
            printer->add_indent();
        }

        (*iter)->accept(*this);

        /// print separator (e.g. comma, space)
        if (!separator.empty() && !utils::is_last(iter, elements)) {
            printer->add_element(separator);
        }

        /// newline at the end of statement
        if (statement) {
            printer->add_newline();
        }

        /// if there are multiple inline comments then we want them to be
        /// contiguous and only last comment should have extra line.
        bool extra_newline = false;
        if (!utils::is_last(iter, elements)) {
            extra_newline = true;
            if ((*iter)->is_line_comment() && (*(iter + 1))->is_line_comment()) {
                extra_newline = false;
            }
        }

        /// program blocks need two newlines except last one
        if (program) {
            printer->add_newline();
            if (extra_newline) {
                printer->add_newline();
            }
        }
    }
}

}  // namespace visitor
}  // namespace nmodl

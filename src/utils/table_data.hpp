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
 * \brief Implement generic table data structure
 */

#include <sstream>
#include <vector>

#include "utils/string_utils.hpp"


namespace nmodl {
namespace utils {

/**
 * @addtogroup utils
 * @{
 */

/**
 * \class TableData
 * \brief Class to construct and pretty-print tabular data
 *
 * This class is used to construct and print tables (like
 * nmodl::symtab::SymbolTable and performance tables).
 */
struct TableData {
    using TableRowType = std::vector<std::string>;

    /// title of the table
    std::string title;

    /// top header/keys
    TableRowType headers;

    /// data
    std::vector<TableRowType> rows;

    /// alignment for every column of data rows
    std::vector<stringutils::text_alignment> alignments;

    void print(int indent = 0) const;

    void print(std::ostream& stream, int indent = 0) const;
};

/** @} */  // end of utils

}  // namespace utils
}  // namespace nmodl

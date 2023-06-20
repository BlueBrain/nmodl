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
 * \brief \copybrief nmodl::printer::NMODLPrinter
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace nmodl {
namespace printer {

/**
 * @addtogroup printer
 * @{
 */

/**
 * \class NMODLPrinter
 * \brief Helper class for printing AST back to NMDOL test
 *
 * NmodlPrintVisitor transforms AST back to NMODL. This class provided common
 * functionality required by visitor to print nmodl ascii file.
 *
 * \todo Implement Printer as base class to avoid duplication code between
 *       JSONPrinter and NMODLPrinter.
 */
class NMODLPrinter {
  private:
    std::ofstream ofs;
    std::streambuf* sbuf = nullptr;
    std::shared_ptr<std::ostream> result;
    size_t indent_level = 0;

  public:
    NMODLPrinter()
        : result(new std::ostream(std::cout.rdbuf())) {}
    NMODLPrinter(std::ostream& stream)
        : result(new std::ostream(stream.rdbuf())) {}
    NMODLPrinter(const std::string& filename);

    ~NMODLPrinter() {
        ofs.close();
    }

    /// print whitespaces for indentation
    void add_indent();

    /// start of new block scope (i.e. start with "{")
    /// and increases indentation level
    void push_level();

    void add_element(const std::string&);
    void add_newline();

    /// end of current block scope (i.e. end with "}")
    /// and decreases indentation level
    void pop_level();
};

/** @} */  // end of printer

}  // namespace printer
}  // namespace nmodl

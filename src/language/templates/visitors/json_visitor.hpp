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

///
/// THIS FILE IS GENERATED AT BUILD TIME AND SHALL NOT BE EDITED.
///

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::JSONVisitor
 */

#include "printer/json_printer.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class JSONVisitor
 * \brief %Visitor for printing AST in JSON format
 *
 * Convert AST into JSON form form using AST visitor. This is used
 * for debugging or visualization purpose.
 */
class JSONVisitor: public ConstAstVisitor {
  private:
    /// json printer
    std::unique_ptr<printer::JSONPrinter> printer;

    /// true if nmodl corresponding to ast node should be added to json
    bool embed_nmodl = false;

  public:
    JSONVisitor()
        : printer(new printer::JSONPrinter()) {}

    JSONVisitor(std::string filename)
        : printer(new printer::JSONPrinter(filename)) {}

    JSONVisitor(std::ostream& ss)
        : printer(new printer::JSONPrinter(ss)) {}

    JSONVisitor& write(const ast::Program& program) {
        visit_program(program);
        return *this;
    }

    JSONVisitor& flush() {
        printer->flush();
        return *this;
    }

    JSONVisitor& compact_json(bool flag) {
        printer->compact_json(flag);
        return *this;
    }

    JSONVisitor& add_nmodl(bool flag) {
        embed_nmodl = flag;
        return *this;
    }

    JSONVisitor& expand_keys(bool flag) {
        printer->expand_keys(flag);
        return *this;
    }

  protected:
    // clang-format off
    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(const ast::{{ node.class_name }}& node) override;
    {% endfor %}
    // clang-format on
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl


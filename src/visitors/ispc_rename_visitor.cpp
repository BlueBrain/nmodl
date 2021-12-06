/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "ispc_rename_visitor.hpp"

namespace nmodl {
namespace visitor {

IspcRenameVisitor::~IspcRenameVisitor() = default;

IspcRenameVisitor::IspcRenameVisitor(const std::shared_ptr<ast::Program>& ast)
    : RenameVisitor(ast, R"(([0-9\.]*d[\-0-9]+)|([0-9\.]+d[\-0-9]*))", "var_", true, true) {}

}  // namespace visitor
}  // namespace nmodl

#include "visitors/semantic_analysis_visitor.hpp"
#include "ast/function_block.hpp"
#include "ast/procedure_block.hpp"
#include "ast/suffix.hpp"
#include "ast/table_statement.hpp"
#include "utils/logger.hpp"

namespace nmodl {
namespace visitor {

bool SemanticAnalysisVisitor::check(const ast::Program& node) {
    check_fail = false;
    visit_program(node);
    return check_fail;
}

void SemanticAnalysisVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    /// <-- This code is for test 1
    in_procedure_function = true;
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    in_procedure_function = false;
    /// -->
}

void SemanticAnalysisVisitor::visit_function_block(const ast::FunctionBlock& node) {
    /// <-- This code is for test 1
    in_procedure_function = true;
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    in_procedure_function = false;
    /// -->
}

void SemanticAnalysisVisitor::visit_table_statement(const ast::TableStatement&) {
    /// <-- This code is for test 1
    if (in_procedure_function && !one_arg_in_procedure_function) {
        logger->critical("SemanticAnalysisVisitor :: The procedure or function containing the TABLE statement should contains exactly one argument.");
        check_fail = true;
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_suffix(const ast::Suffix& node) {
    /// <-- This code is for test 2
    const auto& type = node.get_type()->get_node_name();
    is_point_process = (type == "POINT_PROCESS" || type == "ARTIFICIAL_CELL");
    /// -->
}

void SemanticAnalysisVisitor::visit_destructor_block(const ast::DestructorBlock& node) {
    /// <-- This code is for test 2
    if (!is_point_process) {
        logger->warn("SemanticAnalysisVisitor :: This mod file is not point process but contains a destructor.");
        check_fail = true;
    }
    /// -->
}

}  // namespace visitor
}  // namespace nmodl

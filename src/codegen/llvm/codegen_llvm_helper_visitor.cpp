
/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen_llvm_helper_visitor.hpp"

#include "ast/all.hpp"
#include "codegen/codegen_helper_visitor.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace codegen {

using namespace fmt::literals;

/**
 * \brief Create variable definition statement
 * @param names Name of the variables to be defined
 * @param type Type of the variables
 * @return Statement defining variables
 */
static std::shared_ptr<ast::CodegenVarListStatement> create_variable_statement(
    const std::vector<std::string>& names,
    ast::AstNodeType type) {
    /// create variables for the given name
    ast::CodegenVarVector variables;
    for (const auto& name: names) {
        auto varname = new ast::Name(new ast::String(name));
        variables.emplace_back(new ast::CodegenVar(0, varname));
    }
    auto var_type = new ast::CodegenVarType(type);
    /// construct statement and return it
    return std::make_shared<ast::CodegenVarListStatement>(var_type, variables);
}

/**
 * \brief Create expression for a given NMODL code statement
 * @param code NMODL code statement
 * @return Expression representing given NMODL code
 */
static std::shared_ptr<ast::Expression> create_statement_as_expression(const std::string& code) {
    const auto& statement = visitor::create_statement(code);
    auto expr_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(statement);
    auto expr = expr_statement->get_expression()->clone();
    return std::make_shared<ast::WrappedExpression>(expr);
}

/**
 * \brief Create expression for given NMODL code expression
 * @param code NMODL code expression
 * @return Expression representing NMODL code
 */
std::shared_ptr<ast::Expression> get_expression(const std::string& code) {
    /// as provided code is only expression and not a full statement, create
    /// a temporary assignment statement
    const auto& wrapped_expr = create_statement_as_expression("some_var = " + code);
    /// now extract RHS (representing original code) and return it as expression
    auto expr = std::dynamic_pointer_cast<ast::WrappedExpression>(wrapped_expr)->get_expression();
    auto rhs = std::dynamic_pointer_cast<ast::BinaryExpression>(expr)->get_rhs();
    return std::make_shared<ast::WrappedExpression>(rhs->clone());
}

/**
 * \brief Visit StatementBlock and convert Local statement for code generation
 * @param node AST node representing Statement block
 *
 * Statement blocks can have LOCAL statement and if it exist it's typically
 * first statement in the vector. We have to remove LOCAL statement and convert
 * it to CodegenVarListStatement that will represent all variables as double.
 */
void CodegenLLVMHelperVisitor::visit_statement_block(ast::StatementBlock& node) {
    /// first process all children blocks if any
    node.visit_children(*this);

    /// check if block contains LOCAL statement
    const auto& local_statement = visitor::get_local_list_statement(node);
    if (local_statement) {
        /// create codegen variables from local variables
        ast::CodegenVarVector variables;
        for (const auto& var: local_statement->get_variables()) {
            variables.emplace_back(new ast::CodegenVar(0, var->get_name()->clone()));
        }

        /// remove local list statement now
        const auto& statements = node.get_statements();
        node.erase_statement(statements.begin());

        /// create new codegen variable statement and insert at the beginning of the block
        auto type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
        auto statement = std::make_shared<ast::CodegenVarListStatement>(type, variables);
        node.insert_statement(statements.begin(), statement);
    }
}

/**
 * \brief Add code generation function for FUNCTION or PROCEDURE block
 * @param node AST node representing FUNCTION or PROCEDURE
 *
 * When we have a PROCEDURE or FUNCTION like
 *
 * \code{.mod}
 *      FUNCTION sum(x,y) {
 *          LOCAL res
 *          res = x + y
 *          sum = res
 *      }
 * \endcode
 *
 * this gets typically converted to C/C++ code as:
 *
 * \code{.cpp}
 *      double sum(double x, double y) {
 *          double res;
 *          double ret_sum;
 *          res = x + y;
 *          ret_sum = res;
 *          return ret_sum;
 * \endcode
 *
 * We perform following transformations so that code generation backends
 * will have minimum logic:
 *  - Add return type
 *  - Add type for the function arguments
 *  - Define variables and return variable
 *  - Add return type (int for PROCEDURE and double for FUNCTION)
 */
void CodegenLLVMHelperVisitor::create_function_for_node(ast::Block& node) {
    /// name of the function from the node
    std::string function_name = node.get_node_name();
    auto name = new ast::Name(new ast::String(function_name));

    /// return variable name has "ret_" prefix
    auto return_var = new ast::Name(new ast::String("ret_" + function_name));

    /// return type based on node type
    ast::CodegenVarType* ret_var_type = nullptr;
    if (node.get_node_type() == ast::AstNodeType::FUNCTION_BLOCK) {
        ret_var_type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
    } else {
        ret_var_type = new ast::CodegenVarType(ast::AstNodeType::INTEGER);
    }

    /// function body and it's statement
    auto block = node.get_statement_block()->clone();
    const auto& statements = block->get_statements();

    /// insert return variable at the start of the block
    ast::CodegenVarVector codegen_vars;
    codegen_vars.emplace_back(new ast::CodegenVar(0, return_var->clone()));
    auto statement = std::make_shared<ast::CodegenVarListStatement>(ret_var_type, codegen_vars);
    block->insert_statement(statements.begin(), statement);

    /// add return statement
    auto return_statement = new ast::CodegenReturnStatement(return_var);
    block->emplace_back_statement(return_statement);

    /// prepare function arguments based original node arguments
    ast::CodegenArgumentVector fun_arguments;
    const auto& arguments = node.get_parameters();
    for (const auto& arg: arguments) {
        /// create new type and name for creating new ast node
        auto type = new ast::CodegenVarType(ast::AstNodeType::DOUBLE);
        auto var = arg->get_name()->clone();
        fun_arguments.emplace_back(new ast::CodegenArgument(type, var));
    }

    /// return type of the function is same as return variable type
    ast::CodegenVarType* fun_ret_type = ret_var_type->clone();

    /// we have all information for code generation function, create a new node
    /// which will be inserted later into AST
    auto function =
        std::make_shared<ast::CodegenFunction>(fun_ret_type, name, fun_arguments, block);
    codegen_functions.push_back(function);
}

void CodegenLLVMHelperVisitor::visit_procedure_block(ast::ProcedureBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

void CodegenLLVMHelperVisitor::visit_function_block(ast::FunctionBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

/**
 * \brief Convert ast::NrnStateBlock to corresponding code generation function nrn_state
 * @param node AST node representing ast::NrnStateBlock
 *
 * Solver passes converts DERIVATIVE block from MOD into ast::NrnStateBlock node
 * that represent `nrn_state` function in the generated CPP code. To help this
 * code generation, we perform various transformation on ast::NrnStateBlock and
 * create new code generation function.
 */
void CodegenLLVMHelperVisitor::visit_nrn_state_block(ast::NrnStateBlock& node) {
    /// double and integer variables in the new function
    std::vector<std::string> double_variables{"v"};
    std::vector<std::string> int_variables{"id", "node_id"};

    /// statements for new function to be generated
    ast::StatementVector function_statements;

    /// create variable definition statements and insert at the beginning
    function_statements.push_back(
        create_variable_statement(double_variables, ast::AstNodeType::DOUBLE));
    function_statements.push_back(
        create_variable_statement(int_variables, ast::AstNodeType::INTEGER));

    /// create now main compute part : for loop over channels

    /// for loop constructs : initialization, condition and increment
    const auto& initialization = create_statement_as_expression("id=0");
    const auto& condition = get_expression("id < node_count");
    const auto& increment = create_statement_as_expression("id = id + 1");

    /// loop body : initialization + solve blocks
    ast::StatementVector loop_body;

    /// access node index and corresponding voltage
    loop_body.push_back(visitor::create_statement("node_id = node_index[id]"));
    loop_body.push_back(visitor::create_statement("v = voltage[node_id]"));

    /// extract solution expressions that are derivative blocks
    const auto& solutions = collect_nodes(node, {ast::AstNodeType::SOLUTION_EXPRESSION});
    for (const auto& statement: solutions) {
        const auto& solution = std::dynamic_pointer_cast<ast::SolutionExpression>(statement);
        auto solution_expr = solution->get_node_to_solve()->clone();
        loop_body.emplace_back(std::make_shared<ast::ExpressionStatement>(solution_expr));
    }

    /// now construct a new code block which will become the bidy of the loop
    auto block = std::make_shared<ast::StatementBlock>(loop_body);

    /// create for loop node
    auto for_loop_statement =
        std::make_shared<ast::CodegenForStatement>(initialization, condition, increment, block);

    /// loop itself becomes one of the statement in the function
    function_statements.push_back(for_loop_statement);

    /// new block for the function
    auto function_block = new ast::StatementBlock(function_statements);

    /// name of the function and it's return type
    std::string function_name = "nrn_state_" + stringutils::tolower(info.mod_suffix);
    auto name = new ast::Name(new ast::String(function_name));
    auto return_type = new ast::CodegenVarType(ast::AstNodeType::VOID);

    /// \todo : currently there are no arguments
    ast::CodegenArgumentVector code_arguments;

    /// finally, create new function
    auto function =
        std::make_shared<ast::CodegenFunction>(return_type, name, code_arguments, function_block);
    codegen_functions.push_back(function);

    std::cout << nmodl::to_nmodl(function);
}

void CodegenLLVMHelperVisitor::visit_program(ast::Program& node) {
    /// run codegen helper visitor to collect information
    CodegenHelperVisitor v;
    info = v.analyze(node);

    logger->info("Running CodegenLLVMHelperVisitor");
    node.visit_children(*this);
    for (auto& fun: codegen_functions) {
        node.emplace_back_node(fun);
    }
}

}  // namespace codegen
}  // namespace nmodl

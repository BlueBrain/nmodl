
/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen_llvm_helper_visitor.hpp"

#include "ast/all.hpp"
#include "codegen/codegen_helper_visitor.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/logger.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace codegen {

using namespace fmt::literals;

using symtab::syminfo::Status;

/// initialize static member variables
const ast::AstNodeType CodegenLLVMHelperVisitor::INTEGER_TYPE = ast::AstNodeType::INTEGER;
const ast::AstNodeType CodegenLLVMHelperVisitor::FLOAT_TYPE = ast::AstNodeType::DOUBLE;
const std::string CodegenLLVMHelperVisitor::NODECOUNT_VAR = "node_count";
const std::string CodegenLLVMHelperVisitor::VOLTAGE_VAR = "voltage";
const std::string CodegenLLVMHelperVisitor::NODE_INDEX_VAR = "node_index";

static constexpr const char epilogue_variable_prefix[] = "epilogue_";

/// Create asr::Varname node with given a given variable name
static ast::VarName* create_varname(const std::string& varname) {
    return new ast::VarName(new ast::Name(new ast::String(varname)), nullptr, nullptr);
}

/**
 * Create initialization expression
 * @param code Usually "id = 0" as a string
 * @return Expression representing code
 * \todo : we can not use `create_statement_as_expression` function because
 *         NMODL parser is using `ast::Double` type to represent all variables
 *         including Integer. See #542.
 */
static std::shared_ptr<ast::Expression> int_initialization_expression(
    const std::string& induction_var,
    int value = 0) {
    // create id = 0
    const auto& id = create_varname(induction_var);
    const auto& zero = new ast::Integer(value, nullptr);
    return std::make_shared<ast::BinaryExpression>(id, ast::BinaryOperator(ast::BOP_ASSIGN), zero);
}

/**
 * \brief Create variable definition statement
 *
 * `LOCAL` variables in NMODL don't have type. These variables need
 * to be defined with float type. Same for index, loop iteration and
 * local variables. This helper function function is used to create
 * type specific local variable.
 *
 * @param names Name of the variables to be defined
 * @param type Type of the variables
 * @return Statement defining variables
 */
static std::shared_ptr<ast::CodegenVarListStatement> create_local_variable_statement(
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
std::shared_ptr<ast::Expression> create_expression(const std::string& code) {
    /// as provided code is only expression and not a full statement, create
    /// a temporary assignment statement
    const auto& wrapped_expr = create_statement_as_expression("some_var = " + code);
    /// now extract RHS (representing original code) and return it as expression
    auto expr = std::dynamic_pointer_cast<ast::WrappedExpression>(wrapped_expr)->get_expression();
    auto rhs = std::dynamic_pointer_cast<ast::BinaryExpression>(expr)->get_rhs();
    return std::make_shared<ast::WrappedExpression>(rhs->clone());
}

CodegenFunctionVector CodegenLLVMHelperVisitor::get_codegen_functions(const ast::Program& node) {
    const_cast<ast::Program&>(node).accept(*this);
    return codegen_functions;
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
 *  - Add type for the function arguments
 *  - Define variables and return variable
 *  - Add return type (int for PROCEDURE and double for FUNCTION)
 */
void CodegenLLVMHelperVisitor::create_function_for_node(ast::Block& node) {
    /// name of the function from the node
    std::string function_name = node.get_node_name();
    auto name = new ast::Name(new ast::String(function_name));

    /// return variable name has "ret_" prefix
    std::string return_var_name = "ret_{}"_format(function_name);
    auto return_var = new ast::Name(new ast::String(return_var_name));

    /// return type based on node type
    ast::CodegenVarType* ret_var_type = nullptr;
    if (node.get_node_type() == ast::AstNodeType::FUNCTION_BLOCK) {
        ret_var_type = new ast::CodegenVarType(FLOAT_TYPE);
    } else {
        ret_var_type = new ast::CodegenVarType(INTEGER_TYPE);
    }

    /// function body and it's statement, copy original block
    auto block = node.get_statement_block()->clone();
    const auto& statements = block->get_statements();

    /// convert local statement to codegenvar statement
    convert_local_statement(*block);

    if (node.get_node_type() == ast::AstNodeType::PROCEDURE_BLOCK) {
        block->insert_statement(statements.begin(),
                                std::make_shared<ast::ExpressionStatement>(
                                    int_initialization_expression(return_var_name)));
    }
    /// insert return variable at the start of the block
    ast::CodegenVarVector codegen_vars;
    codegen_vars.emplace_back(new ast::CodegenVar(0, return_var->clone()));
    auto statement = std::make_shared<ast::CodegenVarListStatement>(ret_var_type, codegen_vars);
    block->insert_statement(statements.begin(), statement);

    /// add return statement
    auto return_statement = new ast::CodegenReturnStatement(return_var);
    block->emplace_back_statement(return_statement);

    /// prepare function arguments based original node arguments
    ast::CodegenVarWithTypeVector arguments;
    for (const auto& param: node.get_parameters()) {
        /// create new type and name for creating new ast node
        auto type = new ast::CodegenVarType(FLOAT_TYPE);
        auto var = param->get_name()->clone();
        arguments.emplace_back(new ast::CodegenVarWithType(type, /*is_pointer=*/0, var));
    }

    /// return type of the function is same as return variable type
    ast::CodegenVarType* fun_ret_type = ret_var_type->clone();

    /// we have all information for code generation function, create a new node
    /// which will be inserted later into AST
    auto function = std::make_shared<ast::CodegenFunction>(fun_ret_type, name, arguments, block);
    if (node.get_token()) {
        function->set_token(*node.get_token()->clone());
    }
    codegen_functions.push_back(function);
}

/**
 * \note : Order of variables is not important but we assume all pointers
 * are added first and then scalar variables like t, dt, second_order etc.
 * This order is assumed when we allocate data for integration testing
 * and benchmarking purpose. See CodegenDataHelper::create_data().
 */
std::shared_ptr<ast::InstanceStruct> CodegenLLVMHelperVisitor::create_instance_struct() {
    ast::CodegenVarWithTypeVector codegen_vars;

    auto add_var_with_type =
        [&](const std::string& name, const ast::AstNodeType type, int is_pointer) {
            auto var_name = new ast::Name(new ast::String(name));
            auto var_type = new ast::CodegenVarType(type);
            auto codegen_var = new ast::CodegenVarWithType(var_type, is_pointer, var_name);
            codegen_vars.emplace_back(codegen_var);
        };

    /// float variables are standard pointers to float vectors
    for (const auto& float_var: info.codegen_float_variables) {
        add_var_with_type(float_var->get_name(), FLOAT_TYPE, /*is_pointer=*/1);
    }

    /// int variables are pointers to indexes for other vectors
    for (const auto& int_var: info.codegen_int_variables) {
        add_var_with_type(int_var.symbol->get_name(), FLOAT_TYPE, /*is_pointer=*/1);
    }

    // for integer variables, there should be index
    for (const auto& int_var: info.codegen_int_variables) {
        std::string var_name = int_var.symbol->get_name() + "_index";
        add_var_with_type(var_name, INTEGER_TYPE, /*is_pointer=*/1);
    }

    // add voltage and node index
    add_var_with_type(VOLTAGE_VAR, FLOAT_TYPE, /*is_pointer=*/1);
    add_var_with_type(NODE_INDEX_VAR, INTEGER_TYPE, /*is_pointer=*/1);

    // add dt, t, celsius
    add_var_with_type(naming::NTHREAD_T_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::NTHREAD_DT_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::CELSIUS_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::SECOND_ORDER_VARIABLE, INTEGER_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::MECH_NODECOUNT_VAR, INTEGER_TYPE, /*is_pointer=*/0);

    return std::make_shared<ast::InstanceStruct>(codegen_vars);
}

static void append_statements_from_block(ast::StatementVector& statements,
                                         const std::shared_ptr<ast::StatementBlock>& block) {
    const auto& block_statements = block->get_statements();
    for (const auto& statement: block_statements) {
        const auto& expression_statement = std::dynamic_pointer_cast<ast::ExpressionStatement>(
            statement);
        if (!expression_statement || !expression_statement->get_expression()->is_solve_block())
            statements.push_back(statement);
    }
}

static std::shared_ptr<ast::CodegenAtomicStatement> create_atomic_statement(
    std::string& ion_varname,
    std::string& index_varname,
    std::string& op_str,
    std::string& rhs_str) {
    // create lhs expression
    auto varname = new ast::Name(new ast::String(ion_varname));
    auto index = new ast::Name(new ast::String(index_varname));
    auto lhs = std::make_shared<ast::VarName>(new ast::IndexedName(varname, index),
                                              /*at=*/nullptr,
                                              /*index=*/nullptr);

    auto op = ast::BinaryOperator(ast::string_to_binaryop(op_str));
    auto rhs = create_expression(rhs_str);
    return std::make_shared<ast::CodegenAtomicStatement>(lhs, op, rhs);
}

/**
 * For a given block type, add read ion statements
 *
 * Depending upon the block type, we have to update read ion variables
 * during code generation. Depending on block/procedure being printed,
 * this method adds necessary read ion variable statements and also
 * corresponding index calculation statements. Note that index statements
 * are added separately at the beginning for just readability purpose.
 *
 * @param type The type of code block being generated
 * @param int_variables Index variables to be created
 * @param double_variables Floating point variables to be created
 * @param index_statements Statements for loading indexes (typically for ions)
 * @param body_statements main compute/update statements
 *
 * \todo After looking into mod2c and neuron implementation, it seems like
 * Ode block type is not used. Need to look into implementation details.
 *
 * \todo Ion copy optimization is not implemented yet. This is currently
 * implemented in C backend using `ion_read_statements_optimized()`.
 */
void CodegenLLVMHelperVisitor::ion_read_statements(BlockType type,
                                                   std::vector<std::string>& int_variables,
                                                   std::vector<std::string>& double_variables,
                                                   ast::StatementVector& index_statements,
                                                   ast::StatementVector& body_statements) {
    /// create read ion and corresponding index statements
    auto create_read_statements = [&](std::pair<std::string, std::string> variable_names) {
        // variable in current mechanism instance
        std::string& varname = variable_names.first;
        // ion variable to be read
        std::string& ion_varname = variable_names.second;
        // index for reading ion variable
        std::string index_varname = "{}_id"_format(varname);
        // first load the index
        std::string index_statement = "{} = {}_index[id]"_format(index_varname, ion_varname);
        // now assign the value
        std::string read_statement = "{} = {}[{}]"_format(varname, ion_varname, index_varname);
        // push index definition, index statement and actual read statement
        int_variables.push_back(index_varname);
        index_statements.push_back(visitor::create_statement(index_statement));
        body_statements.push_back(visitor::create_statement(read_statement));
    };

    /// iterate over all ions and create statements for given block type
    for (const auto& ion: info.ions) {
        const std::string& name = ion.name;
        for (const auto& var: ion.reads) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && info.state_variable(var)) {
                continue;
            }
            auto variable_names = info.read_ion_variable_name(var);
            create_read_statements(variable_names);
        }
        for (const auto& var: ion.writes) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && info.state_variable(var)) {
                continue;
            }
            if (ion.is_ionic_conc(var)) {
                auto variable_names = info.read_ion_variable_name(var);
                create_read_statements(variable_names);
            }
        }
    }
}

/**
 * For a given block type, add write ion statements
 *
 * Depending upon the block type, we have to update write ion variables
 * during code generation. Depending on block/procedure being printed,
 * this method adds necessary write ion variable statements and also
 * corresponding index calculation statements. Note that index statements
 * are added separately at the beginning for just readability purpose.
 *
 * @param type The type of code block being generated
 * @param int_variables Index variables to be created
 * @param double_variables Floating point variables to be created
 * @param index_statements Statements for loading indexes (typically for ions)
 * @param body_statements main compute/update statements
 *
 * \todo If intra or extra cellular ionic concentration is written
 * then it requires call to `nrn_wrote_conc`. In C backend this is
 * implemented in `ion_write_statements()` itself but this is not
 * handled yet.
 */
void CodegenLLVMHelperVisitor::ion_write_statements(BlockType type,
                                                    std::vector<std::string>& int_variables,
                                                    std::vector<std::string>& double_variables,
                                                    ast::StatementVector& index_statements,
                                                    ast::StatementVector& body_statements) {
    /// create write ion and corresponding index statements
    auto create_write_statements = [&](std::string ion_varname, std::string op, std::string rhs) {
        // index for writing ion variable
        std::string index_varname = "{}_id"_format(ion_varname);
        // load index
        std::string index_statement = "{} = {}_index[id]"_format(index_varname, ion_varname);
        // push index definition, index statement and actual write statement
        int_variables.push_back(index_varname);
        index_statements.push_back(visitor::create_statement(index_statement));

        // pass ion variable to write and its index

        // \todo: for ionic variable we don't need atomic operation for single threaded
        //        execution. So let's skip this for now. See below comment for details:
        //        https://github.com/BlueBrain/nmodl/pull/645#issuecomment-1095567789
        // body_statements.push_back(create_atomic_statement(ion_varname, index_varname, op, rhs));

        // lhs variable
        std::string lhs = "{}[{}] "_format(ion_varname, index_varname);

        // lets turn a += b into a = a + b if applicable
        // note that this is done in order to facilitate existing implementation in the llvm
        // backend which doesn't support += or -= operators.
        std::string statement;
        if (!op.compare("+=")) {
            statement = "{} = {} + {}"_format(lhs, lhs, rhs);
        } else if (!op.compare("-=")) {
            statement = "{} = {} - {}"_format(lhs, lhs, rhs);
        } else {
            statement = "{} {} {}"_format(lhs, op, rhs);
        }
        body_statements.push_back(visitor::create_statement(statement));
    };

    /// iterate over all ions and create write ion statements for given block type
    for (const auto& ion: info.ions) {
        std::string concentration;
        std::string name = ion.name;
        for (const auto& var: ion.writes) {
            auto variable_names = info.write_ion_variable_name(var);
            /// ionic currents are accumulated
            if (ion.is_ionic_current(var)) {
                if (type == BlockType::Equation) {
                    std::string current = info.breakpoint_current(var);
                    std::string lhs = variable_names.first;
                    std::string op = "+=";
                    std::string rhs = current;
                    // for synapse type
                    if (info.point_process) {
                        auto area = codegen::naming::NODE_AREA_VARIABLE;
                        rhs += "*(1.e2/{})"_format(area);
                    }
                    create_write_statements(lhs, op, rhs);
                }
            } else {
                if (!ion.is_rev_potential(var)) {
                    concentration = var;
                }
                std::string lhs = variable_names.first;
                std::string op = "=";
                std::string rhs = variable_names.second;
                create_write_statements(lhs, op, rhs);
            }
        }

        /// still need to handle, need to define easy to use API
        if (type == BlockType::Initial && !concentration.empty()) {
            int index = 0;
            if (ion.is_intra_cell_conc(concentration)) {
                index = 1;
            } else if (ion.is_extra_cell_conc(concentration)) {
                index = 2;
            } else {
                /// \todo Unhandled case also in neuron implementation
                throw std::logic_error("codegen error for {} ion"_format(ion.name));
            }
            std::string ion_type_name = "{}_type"_format(ion.name);
            std::string lhs = "int {}"_format(ion_type_name);
            std::string op = "=";
            std::string rhs = ion_type_name;
            create_write_statements(lhs, op, rhs);
            logger->error("conc_write_statement() call is required but it's not supported");
        }
    }
}

/**
 * Convert variables in given node to instance variables
 *
 * For code generation, variables of type range, assigned, state or parameter+range
 * needs to be converted to instance variable i.e. they need to be accessed with
 * loop index variable. For example, `h` variables needs to be converted to `h[id]`.
 *
 * @param node Ast node under which variables to be converted to instance type
 */
void CodegenLLVMHelperVisitor::convert_to_instance_variable(ast::Node& node,
                                                            const std::string& index_var) {
    /// collect all variables in the node of type ast::VarName
    auto variables = collect_nodes(node, {ast::AstNodeType::VAR_NAME});
    for (const auto& v: variables) {
        auto variable = std::dynamic_pointer_cast<ast::VarName>(v);
        auto variable_name = variable->get_node_name();

        /// all instance variables defined in the mod file should be converted to
        /// indexed variables based on the loop iteration variable
        if (info.is_an_instance_variable(variable_name)) {
            auto name = variable->get_name()->clone();
            auto index = new ast::Name(new ast::String(index_var));
            auto indexed_name = std::make_shared<ast::IndexedName>(name, index);
            variable->set_name(indexed_name);
        }

        /// instance_var_helper check of instance variables from mod file as well
        /// as extra variables like ion index variables added for code generation
        if (instance_var_helper.is_an_instance_variable(variable_name)) {
            auto name = new ast::Name(new ast::String(naming::MECH_INSTANCE_VAR));
            auto var = std::make_shared<ast::CodegenInstanceVar>(name, variable->clone());
            variable->set_name(var);
        }
    }
}

/**
 * \brief Visit StatementBlock and convert Local statement for code generation
 * @param node AST node representing Statement block
 *
 * Statement blocks can have LOCAL statement and if it exist it's typically
 * first statement in the vector. We have to remove LOCAL statement and convert
 * it to CodegenVarListStatement that will represent all variables as double.
 */
void CodegenLLVMHelperVisitor::convert_local_statement(ast::StatementBlock& node) {
    /// collect all local statement block
    const auto& statements = collect_nodes(node, {ast::AstNodeType::LOCAL_LIST_STATEMENT});

    /// iterate over all statements and replace each with codegen variable
    for (const auto& statement: statements) {
        const auto& local_statement = std::dynamic_pointer_cast<ast::LocalListStatement>(statement);

        /// create codegen variables from local variables
        /// clone variable to make new independent statement
        ast::CodegenVarVector variables;
        for (const auto& var: local_statement->get_variables()) {
            variables.emplace_back(new ast::CodegenVar(0, var->get_name()->clone()));
        }

        /// remove local list statement now
        std::unordered_set<nmodl::ast::Statement*> to_delete({local_statement.get()});
        /// local list statement is enclosed in statement block
        const auto& parent_node = dynamic_cast<ast::StatementBlock*>(local_statement->get_parent());
        parent_node->erase_statement(to_delete);

        /// create new codegen variable statement and insert at the beginning of the block
        auto type = new ast::CodegenVarType(FLOAT_TYPE);
        auto new_statement = std::make_shared<ast::CodegenVarListStatement>(type, variables);
        const auto& statements = parent_node->get_statements();
        parent_node->insert_statement(statements.begin(), new_statement);
    }
}

/**
 * \brief Visit StatementBlock and rename all LOCAL variables
 * @param node AST node representing Statement block
 *
 * Statement block in remainder loop will have same LOCAL variables from
 * main loop. In order to avoid conflict during lookup, rename each local
 * variable by appending unique number. The number used as suffix is just
 * a counter used for Statement block.
 */
void CodegenLLVMHelperVisitor::rename_local_variables(ast::StatementBlock& node) {
    /// local block counter just to append unique number
    static int local_block_counter = 1;

    /// collect all local statement block
    const auto& statements = collect_nodes(node, {ast::AstNodeType::LOCAL_LIST_STATEMENT});

    /// iterate over each statement and rename all variables
    for (const auto& statement: statements) {
        const auto& local_statement = std::dynamic_pointer_cast<ast::LocalListStatement>(statement);

        /// rename local variable in entire statement block
        for (auto& var: local_statement->get_variables()) {
            std::string old_name = var->get_node_name();
            std::string new_name = "{}_{}"_format(old_name, local_block_counter);
            visitor::RenameVisitor(old_name, new_name).visit_statement_block(node);
        }
    }

    /// make it unique for next statement block
    local_block_counter++;
}


void CodegenLLVMHelperVisitor::visit_procedure_block(ast::ProcedureBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

void CodegenLLVMHelperVisitor::visit_function_block(ast::FunctionBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

std::shared_ptr<ast::Expression> CodegenLLVMHelperVisitor::loop_initialization_expression(
    const std::string& induction_var,
    bool is_remainder_loop) {
    if (platform.is_gpu()) {
        const auto& id = create_varname(induction_var);
        const auto& tid = new ast::CodegenThreadId();
        return std::make_shared<ast::BinaryExpression>(id,
                                                       ast::BinaryOperator(ast::BOP_ASSIGN),
                                                       tid);
    }

    // Otherwise, platfrom is CPU. Since the loop can be a remainder loop, check if
    // we need to initialize at all.
    if (is_remainder_loop)
        return nullptr;
    return int_initialization_expression(induction_var);
}

std::shared_ptr<ast::Expression> CodegenLLVMHelperVisitor::loop_increment_expression(
    const std::string& induction_var,
    bool is_remainder_loop) {
    const auto& id = create_varname(induction_var);

    // For GPU platforms, increment by grid stride.
    if (platform.is_gpu()) {
        const auto& stride = new ast::CodegenGridStride();
        const auto& inc_expr =
            new ast::BinaryExpression(id, ast::BinaryOperator(ast::BOP_ADDITION), stride);
        return std::make_shared<ast::BinaryExpression>(id->clone(),
                                                       ast::BinaryOperator(ast::BOP_ASSIGN),
                                                       inc_expr);
    }

    // Otherwise, proceed with increment for CPU loop.
    const int width = is_remainder_loop ? 1 : platform.get_instruction_width();
    const auto& inc = new ast::Integer(width, nullptr);
    const auto& inc_expr =
        new ast::BinaryExpression(id, ast::BinaryOperator(ast::BOP_ADDITION), inc);
    return std::make_shared<ast::BinaryExpression>(id->clone(),
                                                   ast::BinaryOperator(ast::BOP_ASSIGN),
                                                   inc_expr);
}

std::shared_ptr<ast::Expression> CodegenLLVMHelperVisitor::loop_count_expression(
    const std::string& induction_var,
    const std::string& node_count,
    bool is_remainder_loop) {
    const int width = is_remainder_loop ? 1 : platform.get_instruction_width();
    const auto& id = create_varname(induction_var);
    const auto& mech_node_count = create_varname(node_count);

    // For non-vectorised loop, the condition is id < mech->node_count
    if (width == 1) {
        return std::make_shared<ast::BinaryExpression>(id->clone(),
                                                       ast::BinaryOperator(ast::BOP_LESS),
                                                       mech_node_count);
    }

    // For vectorised loop, the condition is id < mech->node_count - width + 1
    const auto& remainder = new ast::Integer(width - 1, /*macro=*/nullptr);
    const auto& count = new ast::BinaryExpression(mech_node_count,
                                                  ast::BinaryOperator(ast::BOP_SUBTRACTION),
                                                  remainder);
    return std::make_shared<ast::BinaryExpression>(id->clone(),
                                                   ast::BinaryOperator(ast::BOP_LESS),
                                                   count);
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
    /// statements for new function to be generated
    ast::StatementVector function_statements;

    /// create vectors of local variables that would be used in compute part
    std::vector<std::string> int_variables{"node_id"};
    std::vector<std::string> double_variables{"v"};

    /// create now main compute part

    /// compute body : initialization + solve blocks
    ast::StatementVector def_statements;
    ast::StatementVector index_statements;
    ast::StatementVector body_statements;
    {
        /// access node index and corresponding voltage
        index_statements.push_back(
            visitor::create_statement("node_id = node_index[{}]"_format(naming::INDUCTION_VAR)));
        body_statements.push_back(visitor::create_statement("v = {}[node_id]"_format(VOLTAGE_VAR)));

        /// read ion variables
        ion_read_statements(
            BlockType::State, int_variables, double_variables, index_statements, body_statements);

        /// main compute node : extract solution expressions from the derivative block
        const auto& solutions = collect_nodes(node, {ast::AstNodeType::SOLUTION_EXPRESSION});
        for (const auto& statement: solutions) {
            const auto& solution = std::dynamic_pointer_cast<ast::SolutionExpression>(statement);
            const auto& block = std::dynamic_pointer_cast<ast::StatementBlock>(
                solution->get_node_to_solve());
            append_statements_from_block(body_statements, block);
        }

        /// add breakpoint block if no current
        if (info.currents.empty() && info.breakpoint_node != nullptr) {
            auto block = info.breakpoint_node->get_statement_block();
            append_statements_from_block(body_statements, block);
        }

        /// write ion statements
        ion_write_statements(
            BlockType::State, int_variables, double_variables, index_statements, body_statements);

        // \todo handle process_shadow_update_statement and wrote_conc_call yet
    }

    /// create target-specific compute body
    ast::StatementVector compute_body;
    compute_body.insert(compute_body.end(), def_statements.begin(), def_statements.end());
    compute_body.insert(compute_body.end(), index_statements.begin(), index_statements.end());
    compute_body.insert(compute_body.end(), body_statements.begin(), body_statements.end());

    std::vector<std::string> induction_variables{naming::INDUCTION_VAR};
    function_statements.push_back(
        create_local_variable_statement(induction_variables, INTEGER_TYPE));

    if (platform.is_gpu()) {
        create_gpu_compute_body(compute_body, function_statements, int_variables, double_variables);
    } else {
        create_cpu_compute_body(compute_body, function_statements, int_variables, double_variables);
    }

    /// new block for the function
    auto function_block = new ast::StatementBlock(function_statements);

    /// name of the function and it's return type
    std::string function_name = "nrn_state_" + stringutils::tolower(info.mod_suffix);
    auto name = new ast::Name(new ast::String(function_name));
    auto return_type = new ast::CodegenVarType(ast::AstNodeType::VOID);

    /// \todo : currently there are no arguments
    ast::CodegenVarWithTypeVector code_arguments;

    auto instance_var_type = new ast::CodegenVarType(ast::AstNodeType::INSTANCE_STRUCT);
    auto instance_var_name = new ast::Name(new ast::String(naming::MECH_INSTANCE_VAR));
    auto instance_var = new ast::CodegenVarWithType(instance_var_type, 1, instance_var_name);
    code_arguments.emplace_back(instance_var);

    /// finally, create new function
    auto function =
        std::make_shared<ast::CodegenFunction>(return_type, name, code_arguments, function_block);
    codegen_functions.push_back(function);

    std::cout << nmodl::to_nmodl(function) << std::endl;
}

void CodegenLLVMHelperVisitor::create_gpu_compute_body(ast::StatementVector& body,
                                                       ast::StatementVector& function_statements,
                                                       std::vector<std::string>& int_variables,
                                                       std::vector<std::string>& double_variables) {
    auto kernel_block = std::make_shared<ast::StatementBlock>(body);

    // dispatch loop creation with right parameters
    create_compute_body_loop(kernel_block, function_statements, int_variables, double_variables);
}

void CodegenLLVMHelperVisitor::create_cpu_compute_body(ast::StatementVector& body,
                                                       ast::StatementVector& function_statements,
                                                       std::vector<std::string>& int_variables,
                                                       std::vector<std::string>& double_variables) {
    auto loop_block = std::make_shared<ast::StatementBlock>(body);
    create_compute_body_loop(loop_block, function_statements, int_variables, double_variables);
    if (platform.is_cpu_with_simd())
        create_compute_body_loop(loop_block,
                                 function_statements,
                                 int_variables,
                                 double_variables,
                                 /*is_remainder_loop=*/true);
}

void CodegenLLVMHelperVisitor::create_compute_body_loop(std::shared_ptr<ast::StatementBlock>& block,
                                                        ast::StatementVector& function_statements,
                                                        std::vector<std::string>& int_variables,
                                                        std::vector<std::string>& double_variables,
                                                        bool is_remainder_loop) {
    const auto& initialization = loop_initialization_expression(naming::INDUCTION_VAR,
                                                                is_remainder_loop);
    const auto& condition =
        loop_count_expression(naming::INDUCTION_VAR, NODECOUNT_VAR, is_remainder_loop);
    const auto& increment = loop_increment_expression(naming::INDUCTION_VAR, is_remainder_loop);

    // Clone the statement block if needed since it can be used by the remainder loop.
    auto loop_block = (is_remainder_loop || !platform.is_cpu_with_simd())
                          ? block
                          : std::shared_ptr<ast::StatementBlock>(block->clone());

    // Convert local statement to use CodegenVar statements and create a FOR loop node. Also, if
    // creating a remainder loop then rename variables to avoid conflicts.
    if (is_remainder_loop)
        rename_local_variables(*loop_block);
    convert_local_statement(*loop_block);
    auto for_loop = std::make_shared<ast::CodegenForStatement>(initialization,
                                                               condition,
                                                               increment,
                                                               loop_block);

    // Convert all variables inside loop body to be instance variables.
    convert_to_instance_variable(*for_loop, naming::INDUCTION_VAR);

    // Rename variables if processing remainder loop.
    if (is_remainder_loop) {
        const auto& loop_statements = for_loop->get_statement_block();
        auto rename = [&](std::vector<std::string>& vars) {
            for (int i = 0; i < vars.size(); ++i) {
                std::string old_name = vars[i];
                std::string new_name = epilogue_variable_prefix + vars[i];
                vars[i] = new_name;
                visitor::RenameVisitor v(old_name, new_name);
                loop_statements->accept(v);
            }
        };
        rename(int_variables);
        rename(double_variables);
    }

    // Push variables and  the loop to the function statements vector.
    function_statements.push_back(create_local_variable_statement(int_variables, INTEGER_TYPE));
    function_statements.push_back(create_local_variable_statement(double_variables, FLOAT_TYPE));
    function_statements.push_back(for_loop);
}

void CodegenLLVMHelperVisitor::remove_inlined_nodes(ast::Program& node) {
    auto program_symtab = node.get_model_symbol_table();
    const auto& func_proc_nodes =
        collect_nodes(node, {ast::AstNodeType::FUNCTION_BLOCK, ast::AstNodeType::PROCEDURE_BLOCK});
    std::unordered_set<ast::Node*> nodes_to_erase;
    for (const auto& ast_node: func_proc_nodes) {
        if (program_symtab->lookup(ast_node->get_node_name())
                .get()
                ->has_all_status(Status::inlined)) {
            nodes_to_erase.insert(static_cast<ast::Node*>(ast_node.get()));
        }
    }
    node.erase_node(nodes_to_erase);
}

/**
 * \brief Convert ast::BreakpointBlock to corresponding code generation function nrn_cur
 * @param node AST node representing ast::BreakpointBlock
 *
 */
void CodegenLLVMHelperVisitor::visit_breakpoint_block(ast::BreakpointBlock& node) {
    if (!info.nrn_cur_required()) {
        return;
    }

    /// statements for new function to be generated
    ast::StatementVector function_statements;

    /// create vectors of local variables that would be used in compute part
    std::vector<std::string> int_variables{"node_id"};
    std::vector<std::string> double_variables{"v"};

    /// create now main compute part

    /// compute body : initialization + solve blocks
    ast::StatementVector def_statements;
    ast::StatementVector index_statements;
    ast::StatementVector body_statements;
    {
        /// access node index and corresponding voltage
        index_statements.push_back(
            visitor::create_statement("node_id = node_index[{}]"_format(naming::INDUCTION_VAR)));
        body_statements.push_back(visitor::create_statement("v = {}[node_id]"_format(VOLTAGE_VAR)));

        /// read ion variables
        ion_read_statements(BlockType::Equation,
                            int_variables,
                            double_variables,
                            index_statements,
                            body_statements);

        /// main compute node : extract solution expressions from the derivative block
        const auto& block = node.get_statement_block();
        append_statements_from_block(body_statements, block);

        /// write ion statements
        ion_write_statements(BlockType::Equation,
                             int_variables,
                             double_variables,
                             index_statements,
                             body_statements);

        // \todo handle nrn_current calls similar to C codegen backend
    }

    /// create target-specific compute body
    ast::StatementVector compute_body;
    compute_body.insert(compute_body.end(), def_statements.begin(), def_statements.end());
    compute_body.insert(compute_body.end(), index_statements.begin(), index_statements.end());
    compute_body.insert(compute_body.end(), body_statements.begin(), body_statements.end());

    std::vector<std::string> induction_variables{naming::INDUCTION_VAR};
    function_statements.push_back(
        create_local_variable_statement(induction_variables, INTEGER_TYPE));

    if (platform.is_gpu()) {
        create_gpu_compute_body(compute_body, function_statements, int_variables, double_variables);
    } else {
        create_cpu_compute_body(compute_body, function_statements, int_variables, double_variables);
    }

    /// new block for the function
    auto function_block = new ast::StatementBlock(function_statements);

    /// name of the function and it's return type
    std::string function_name = "nrn_cur_" + stringutils::tolower(info.mod_suffix);
    auto name = new ast::Name(new ast::String(function_name));
    auto return_type = new ast::CodegenVarType(ast::AstNodeType::VOID);

    ast::CodegenVarWithTypeVector code_arguments;

    auto instance_var_type = new ast::CodegenVarType(ast::AstNodeType::INSTANCE_STRUCT);
    auto instance_var_name = new ast::Name(new ast::String(naming::MECH_INSTANCE_VAR));
    auto instance_var = new ast::CodegenVarWithType(instance_var_type, 1, instance_var_name);
    code_arguments.emplace_back(instance_var);

    /// finally, create new function
    auto function =
        std::make_shared<ast::CodegenFunction>(return_type, name, code_arguments, function_block);
    codegen_functions.push_back(function);

    std::cout << nmodl::to_nmodl(function) << std::endl;
}

void CodegenLLVMHelperVisitor::visit_program(ast::Program& node) {
    /// run codegen helper visitor to collect information
    CodegenHelperVisitor v;
    info = v.analyze(node);

    instance_var_helper.instance = create_instance_struct();
    node.emplace_back_node(instance_var_helper.instance);

    logger->info("Running CodegenLLVMHelperVisitor");
    remove_inlined_nodes(node);
    node.visit_children(*this);
    for (auto& fun: codegen_functions) {
        node.emplace_back_node(fun);
    }
}


}  // namespace codegen
}  // namespace nmodl


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
    auto return_var = new ast::Name(new ast::String("ret_" + function_name));

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
    codegen_functions.push_back(function);
}

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
    add_var_with_type("voltage", FLOAT_TYPE, /*is_pointer=*/1);
    add_var_with_type("node_index", INTEGER_TYPE, /*is_pointer=*/1);

    // add dt, t, celsius
    add_var_with_type(naming::NTHREAD_T_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::NTHREAD_DT_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::CELSIUS_VARIABLE, FLOAT_TYPE, /*is_pointer=*/0);
    add_var_with_type(naming::SECOND_ORDER_VARIABLE, INTEGER_TYPE, /*is_pointer=*/0);
    add_var_with_type(MECH_NODECOUNT_VAR, INTEGER_TYPE, /*is_pointer=*/0);

    return std::make_shared<ast::InstanceStruct>(codegen_vars);
}

static void append_statements_from_block(ast::StatementVector& statements,
                                         const std::shared_ptr<ast::StatementBlock>& block) {
    const auto& block_statements = block->get_statements();
    statements.insert(statements.end(), block_statements.begin(), block_statements.end());
}

static std::shared_ptr<ast::CodegenAtomicStatement> create_atomic_statement(std::string& lhs_str,
                                                                            std::string& op_str,
                                                                            std::string& rhs_str) {
    auto lhs = std::make_shared<ast::Name>(new ast::String(lhs_str));
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
        // ion variable to write (with index)
        std::string ion_to_write = "{}[{}]"_format(ion_varname, index_varname);
        // push index definition, index statement and actual write statement
        int_variables.push_back(index_varname);
        index_statements.push_back(visitor::create_statement(index_statement));
        body_statements.push_back(create_atomic_statement(ion_to_write, op, rhs));
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
                                                            std::string& index_var) {
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
            auto name = new ast::Name(new ast::String(MECH_INSTANCE_VAR));
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
    /// first process all children blocks if any
    node.visit_children(*this);

    /// check if block contains LOCAL statement
    const auto& local_statement = visitor::get_local_list_statement(node);
    if (local_statement) {
        /// create codegen variables from local variables
        /// clone variable to make new independent statement
        ast::CodegenVarVector variables;
        for (const auto& var: local_statement->get_variables()) {
            variables.emplace_back(new ast::CodegenVar(0, var->get_name()->clone()));
        }

        /// remove local list statement now
        const auto& statements = node.get_statements();
        node.erase_statement(statements.begin());

        /// create new codegen variable statement and insert at the beginning of the block
        auto type = new ast::CodegenVarType(FLOAT_TYPE);
        auto statement = std::make_shared<ast::CodegenVarListStatement>(type, variables);
        node.insert_statement(statements.begin(), statement);
    }
}

void CodegenLLVMHelperVisitor::visit_procedure_block(ast::ProcedureBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

void CodegenLLVMHelperVisitor::visit_function_block(ast::FunctionBlock& node) {
    node.visit_children(*this);
    create_function_for_node(node);
}

/// Create asr::Varname node with given a given variable name
static ast::VarName* create_varname(const std::string& varname) {
    return new ast::VarName(new ast::Name(new ast::String(varname)), nullptr, nullptr);
}

/**
 * Create for loop initialization expression
 * @param code Usually "id = 0" as a string
 * @return Expression representing code
 * \todo : we can not use `create_statement_as_expression` function because
 *         NMODL parser is using `ast::Double` type to represent all variables
 *         including Integer. See #542.
 */
static std::shared_ptr<ast::Expression> loop_initialization_expression(
    const std::string& induction_var) {
    // create id = 0
    const auto& id = create_varname(induction_var);
    const auto& zero = new ast::Integer(0, nullptr);
    return std::make_shared<ast::BinaryExpression>(id, ast::BinaryOperator(ast::BOP_ASSIGN), zero);
}

/**
 * Create loop increment expression `id = id + width`
 * \todo : same as loop_initialization_expression()
 */
static std::shared_ptr<ast::Expression> loop_increment_expression(const std::string& induction_var,
                                                                  int vector_width) {
    // first create id + x
    const auto& id = create_varname(induction_var);
    const auto& inc = new ast::Integer(vector_width, nullptr);
    const auto& inc_expr =
        new ast::BinaryExpression(id, ast::BinaryOperator(ast::BOP_ADDITION), inc);
    // now create id = id + x
    return std::make_shared<ast::BinaryExpression>(id->clone(),
                                                   ast::BinaryOperator(ast::BOP_ASSIGN),
                                                   inc_expr);
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

    /// create variable definition for loop index and insert at the beginning
    std::string loop_index_var = "id";
    std::vector<std::string> int_variables{"id"};
    function_statements.push_back(create_local_variable_statement(int_variables, INTEGER_TYPE));

    /// create now main compute part : for loop over channel instances

    /// loop constructs : initialization, condition and increment
    const auto& initialization = loop_initialization_expression(INDUCTION_VAR);
    const auto& condition = create_expression("{} < {}"_format(INDUCTION_VAR, MECH_NODECOUNT_VAR));
    const auto& increment = loop_increment_expression(INDUCTION_VAR, vector_width);

    /// loop body : initialization + solve blocks
    ast::StatementVector loop_def_statements;
    ast::StatementVector loop_index_statements;
    ast::StatementVector loop_body_statements;
    {
        std::vector<std::string> int_variables{"node_id"};
        std::vector<std::string> double_variables{"v"};

        /// access node index and corresponding voltage
        loop_index_statements.push_back(
            visitor::create_statement("node_id = node_index[{}]"_format(INDUCTION_VAR)));
        loop_body_statements.push_back(visitor::create_statement("v = voltage[node_id]"));

        /// read ion variables
        ion_read_statements(BlockType::State,
                            int_variables,
                            double_variables,
                            loop_index_statements,
                            loop_body_statements);

        /// main compute node : extract solution expressions from the derivative block
        const auto& solutions = collect_nodes(node, {ast::AstNodeType::SOLUTION_EXPRESSION});
        for (const auto& statement: solutions) {
            const auto& solution = std::dynamic_pointer_cast<ast::SolutionExpression>(statement);
            const auto& block = std::dynamic_pointer_cast<ast::StatementBlock>(
                solution->get_node_to_solve());
            append_statements_from_block(loop_body_statements, block);
        }

        /// add breakpoint block if no current
        if (info.currents.empty() && info.breakpoint_node != nullptr) {
            auto block = info.breakpoint_node->get_statement_block();
            append_statements_from_block(loop_body_statements, block);
        }

        /// write ion statements
        ion_write_statements(BlockType::State,
                             int_variables,
                             double_variables,
                             loop_index_statements,
                             loop_body_statements);

        loop_def_statements.push_back(create_local_variable_statement(int_variables, INTEGER_TYPE));
        loop_def_statements.push_back(
            create_local_variable_statement(double_variables, FLOAT_TYPE));

        // \todo handle process_shadow_update_statement and wrote_conc_call yet
    }

    ast::StatementVector loop_body;
    loop_body.insert(loop_body.end(), loop_def_statements.begin(), loop_def_statements.end());
    loop_body.insert(loop_body.end(), loop_index_statements.begin(), loop_index_statements.end());
    loop_body.insert(loop_body.end(), loop_body_statements.begin(), loop_body_statements.end());

    /// now construct a new code block which will become the body of the loop
    auto loop_block = std::make_shared<ast::StatementBlock>(loop_body);

    /// convert local statement to codegenvar statement
    convert_local_statement(*loop_block);

    /// create for loop node
    auto for_loop_statement = std::make_shared<ast::CodegenForStatement>(initialization,
                                                                         condition,
                                                                         increment,
                                                                         loop_block);

    /// convert all variables inside loop body to instance variables
    convert_to_instance_variable(*for_loop_statement, loop_index_var);

    /// loop itself becomes one of the statement in the function
    function_statements.push_back(for_loop_statement);

    /// new block for the function
    auto function_block = new ast::StatementBlock(function_statements);

    /// name of the function and it's return type
    std::string function_name = "nrn_state_" + stringutils::tolower(info.mod_suffix);
    auto name = new ast::Name(new ast::String(function_name));
    auto return_type = new ast::CodegenVarType(ast::AstNodeType::VOID);

    /// \todo : currently there are no arguments
    ast::CodegenVarWithTypeVector code_arguments;

    auto instance_var_type = new ast::CodegenVarType(ast::AstNodeType::INSTANCE_STRUCT);
    auto instance_var_name = new ast::Name(new ast::String(MECH_INSTANCE_VAR));
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
    node.visit_children(*this);
    for (auto& fun: codegen_functions) {
        node.emplace_back_node(fun);
    }
}


}  // namespace codegen
}  // namespace nmodl

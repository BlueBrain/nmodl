/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/llvm_utils.hpp"

#include "ast/all.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/visitor_utils.hpp"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Host.h"

#if LLVM_VERSION_MAJOR >= 13
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#endif

namespace nmodl {
namespace codegen {


static constexpr const char instance_struct_type_name[] = "__instance_var__type";


/****************************************************************************************/
/*                                  Helper routines                                     */
/****************************************************************************************/

/// A utility to check for supported Statement AST nodes.
static bool is_supported_statement(const ast::Statement& statement) {
    return statement.is_codegen_atomic_statement() || statement.is_codegen_for_statement() ||
           statement.is_if_statement() || statement.is_codegen_return_statement() ||
           statement.is_codegen_var_list_statement() || statement.is_expression_statement() ||
           statement.is_while_statement();
}

/// A utility to check that the kernel body can be vectorised.
static bool can_vectorize(const ast::CodegenForStatement& statement, symtab::SymbolTable* sym_tab) {
    // Check that function calls are made to external methods only.
    const auto& function_calls = collect_nodes(statement, {ast::AstNodeType::FUNCTION_CALL});
    for (const auto& call: function_calls) {
        const auto& name = call->get_node_name();
        auto symbol = sym_tab->lookup(name);
        if (symbol && !symbol->has_any_property(symtab::syminfo::NmodlType::extern_method))
            return false;
    }

    // Check for simple supported control flow in the kernel (single if/else statement).
    const std::vector<ast::AstNodeType> supported_control_flow = {ast::AstNodeType::IF_STATEMENT};
    const auto& supported = collect_nodes(statement, supported_control_flow);

    // Check for unsupported control flow statements.
    const std::vector<ast::AstNodeType> unsupported_nodes = {ast::AstNodeType::ELSE_IF_STATEMENT};
    const auto& unsupported = collect_nodes(statement, unsupported_nodes);

    return unsupported.empty() && supported.size() <= 1;
}

#if LLVM_VERSION_MAJOR >= 13
void CodegenLLVMVisitor::add_vectorizable_functions_from_vec_lib(llvm::TargetLibraryInfoImpl& tli,
                                                                 llvm::Triple& triple) {
    // Since LLVM does not support SLEEF as a vector library yet, process it separately.
    if (vector_library == "SLEEF") {
// clang-format off
#define FIXED(w) llvm::ElementCount::getFixed(w)
// clang-format on
#define DISPATCH(func, vec_func, width) {func, vec_func, width},

        // Populate function definitions of only exp and pow (for now)
        const llvm::VecDesc aarch64_functions[] = {
            // clang-format off
            DISPATCH("llvm.exp.f32", "_ZGVnN4v_expf", FIXED(4))
            DISPATCH("llvm.exp.f64", "_ZGVnN2v_exp", FIXED(2))
            DISPATCH("llvm.pow.f32", "_ZGVnN4vv_powf", FIXED(4))
            DISPATCH("llvm.pow.f64", "_ZGVnN2vv_pow", FIXED(2))
            // clang-format on
        };
        const llvm::VecDesc x86_functions[] = {
            // clang-format off
            DISPATCH("llvm.exp.f64", "_ZGVbN2v_exp", FIXED(2))
            DISPATCH("llvm.exp.f64", "_ZGVdN4v_exp", FIXED(4))
            DISPATCH("llvm.exp.f64", "_ZGVeN8v_exp", FIXED(8))
            DISPATCH("llvm.pow.f64", "_ZGVbN2vv_pow", FIXED(2))
            DISPATCH("llvm.pow.f64", "_ZGVdN4vv_pow", FIXED(4))
            DISPATCH("llvm.pow.f64", "_ZGVeN8vv_pow", FIXED(8))
            // clang-format on
        };
#undef DISPATCH

        if (triple.isAArch64()) {
            tli.addVectorizableFunctions(aarch64_functions);
        }
        if (triple.isX86() && triple.isArch64Bit()) {
            tli.addVectorizableFunctions(x86_functions);
        }

    } else {
        // A map to query vector library by its string value.
        using VecLib = llvm::TargetLibraryInfoImpl::VectorLibrary;
        static const std::map<std::string, VecLib> llvm_supported_vector_libraries = {
            {"Accelerate", VecLib::Accelerate},
            {"libmvec", VecLib::LIBMVEC_X86},
            {"libsystem_m", VecLib ::DarwinLibSystemM},
            {"MASSV", VecLib::MASSV},
            {"none", VecLib::NoLibrary},
            {"SVML", VecLib::SVML}};
        const auto& library = llvm_supported_vector_libraries.find(vector_library);
        if (library == llvm_supported_vector_libraries.end())
            throw std::runtime_error("Error: unknown vector library - " + vector_library + "\n");

        // Add vectorizable functions to the target library info.
        switch (library->second) {
        case VecLib::LIBMVEC_X86:
            if (!triple.isX86() || !triple.isArch64Bit())
                break;
        default:
            tli.addVectorizableFunctionsFromVecLib(library->second);
            break;
        }
    }
}
#endif

llvm::Value* CodegenLLVMVisitor::accept_and_get(const std::shared_ptr<ast::Node>& node) {
    node->accept(*this);
    return ir_builder.pop_last_value();
}

void CodegenLLVMVisitor::create_external_function_call(const std::string& name,
                                                       const ast::ExpressionVector& arguments) {
    if (name == "printf") {
        create_printf_call(arguments);
        return;
    }

    ValueVector argument_values;
    TypeVector argument_types;
    for (const auto& arg: arguments) {
        llvm::Value* value = accept_and_get(arg);
        llvm::Type* type = value->getType();
        argument_types.push_back(type);
        argument_values.push_back(value);
    }
    ir_builder.create_intrinsic(name, argument_values, argument_types);
}

void CodegenLLVMVisitor::create_function_call(llvm::Function* func,
                                              const std::string& name,
                                              const ast::ExpressionVector& arguments) {
    // Check that function is called with the expected number of arguments.
    if (!func->isVarArg() && arguments.size() != func->arg_size()) {
        throw std::runtime_error("Error: Incorrect number of arguments passed");
    }

    // Pack function call arguments to vector and create a call instruction.
    ValueVector argument_values;
    argument_values.reserve(arguments.size());
    create_function_call_arguments(arguments, argument_values);
    ir_builder.create_function_call(func, argument_values);
}

void CodegenLLVMVisitor::create_function_call_arguments(const ast::ExpressionVector& arguments,
                                                        ValueVector& arg_values) {
    for (const auto& arg: arguments) {
        if (arg->is_string()) {
            // If the argument is a string, create a global i8* variable with it.
            auto string_arg = std::dynamic_pointer_cast<ast::String>(arg);
            arg_values.push_back(ir_builder.create_global_string(*string_arg));
        } else {
            llvm::Value* value = accept_and_get(arg);
            arg_values.push_back(value);
        }
    }
}

void CodegenLLVMVisitor::create_function_declaration(const ast::CodegenFunction& node) {
    const auto& name = node.get_node_name();
    const auto& arguments = node.get_arguments();

    // Procedure or function parameters are doubles by default.
    TypeVector arg_types;
    for (size_t i = 0; i < arguments.size(); ++i)
        arg_types.push_back(get_codegen_var_type(*arguments[i]->get_type()));

    llvm::Type* return_type = get_codegen_var_type(*node.get_return_type());

    // Create a function that is automatically inserted into module's symbol table.
    auto func =
        llvm::Function::Create(llvm::FunctionType::get(return_type, arg_types, /*isVarArg=*/false),
                               llvm::Function::ExternalLinkage,
                               name,
                               *module);

    // Add function debug information, with location information if it exists.
    if (add_debug_information) {
        if (node.get_token()) {
            Location loc{node.get_token()->start_line(), node.get_token()->start_column()};
            debug_builder.add_function_debug_info(func, &loc);
        } else {
            debug_builder.add_function_debug_info(func);
        }
    }
}

void CodegenLLVMVisitor::create_printf_call(const ast::ExpressionVector& arguments) {
    // First, create printf declaration or insert it if it does not exit.
    std::string name = "printf";
    llvm::Function* printf = module->getFunction(name);
    if (!printf) {
        llvm::FunctionType* printf_type = llvm::FunctionType::get(ir_builder.get_i32_type(),
                                                                  ir_builder.get_i8_ptr_type(),
                                                                  /*isVarArg=*/true);

        printf =
            llvm::Function::Create(printf_type, llvm::Function::ExternalLinkage, name, *module);
    }

    // Create a call instruction.
    ValueVector argument_values;
    argument_values.reserve(arguments.size());
    create_function_call_arguments(arguments, argument_values);
    ir_builder.create_function_call(printf, argument_values, /*use_result=*/false);
}

void CodegenLLVMVisitor::create_vectorized_control_flow_block(const ast::IfStatement& node) {
    // Get the true mask from the condition statement.
    llvm::Value* true_mask = accept_and_get(node.get_condition());

    // Process the true block.
    ir_builder.set_mask(true_mask);
    node.get_statement_block()->accept(*this);

    // Note: by default, we do not support kernels with complicated control flow. This is checked
    // prior to visiting 'CodegenForStatement`.
    const auto& elses = node.get_elses();
    if (elses) {
        // If `else` statement exists, invert the mask and proceed with code generation.
        ir_builder.invert_mask();
        elses->get_statement_block()->accept(*this);
    }

    // Clear the mask value.
    ir_builder.clear_mask();
}

void CodegenLLVMVisitor::find_kernel_names(std::vector<std::string>& container) {
    auto& functions = module->getFunctionList();
    for (auto& func: functions) {
        const std::string name = func.getName().str();
        if (is_kernel_function(name)) {
            container.push_back(name);
        }
    }
}

llvm::Type* CodegenLLVMVisitor::get_codegen_var_type(const ast::CodegenVarType& node) {
    switch (node.get_type()) {
    case ast::AstNodeType::BOOLEAN:
        return ir_builder.get_boolean_type();
    case ast::AstNodeType::DOUBLE:
        return ir_builder.get_fp_type();
    case ast::AstNodeType::INSTANCE_STRUCT:
        return get_instance_struct_type();
    case ast::AstNodeType::INTEGER:
        return ir_builder.get_i32_type();
    case ast::AstNodeType::VOID:
        return ir_builder.get_void_type();
    default:
        throw std::runtime_error("Error: expecting a type in CodegenVarType node\n");
    }
}

llvm::Value* CodegenLLVMVisitor::get_index(const ast::IndexedName& node) {
    // In NMODL, the index is either an integer expression or a named constant, such as "id".
    llvm::Value* index_value = node.get_length()->is_name()
                                   ? ir_builder.create_load(node.get_length()->get_node_name())
                                   : accept_and_get(node.get_length());
    return ir_builder.create_index(index_value);
}

llvm::Type* CodegenLLVMVisitor::get_instance_struct_type() {
    TypeVector member_types;
    for (const auto& variable: instance_var_helper.instance->get_codegen_vars()) {
        // Get the type information of the codegen variable.
        const auto& is_pointer = variable->get_is_pointer();
        const auto& nmodl_type = variable->get_type()->get_type();

        // Create the corresponding LLVM type.
        switch (nmodl_type) {
        case ast::AstNodeType::DOUBLE:
            member_types.push_back(is_pointer ? ir_builder.get_fp_ptr_type()
                                              : ir_builder.get_fp_type());
            break;
        case ast::AstNodeType::INTEGER:
            member_types.push_back(is_pointer ? ir_builder.get_i32_ptr_type()
                                              : ir_builder.get_i32_type());
            break;
        default:
            throw std::runtime_error("Error: unsupported type found in instance struct\n");
        }
    }

    return ir_builder.get_struct_ptr_type(mod_filename + instance_struct_type_name, member_types);
}

int CodegenLLVMVisitor::get_num_elements(const ast::IndexedName& node) {
    // First, verify if the length is an integer value.
    const auto& integer = std::dynamic_pointer_cast<ast::Integer>(node.get_length());
    if (!integer)
        throw std::runtime_error("Error: only integer length is supported\n");

    // Check if the length value is a constant.
    if (!integer->get_macro())
        return integer->get_value();

    // Otherwise, the length is taken from the macro.
    const auto& macro = program_symtab->lookup(integer->get_macro()->get_node_name());
    return static_cast<int>(*macro->get_value());
}

/**
 * Currently, functions are identified as compute kernels if they satisfy the following:
 *   1. They have a void return type
 *   2. They have a single argument
 *   3. The argument is a struct type pointer
 * This is not robust, and hence it would be better to find what functions are kernels on the NMODL
 * AST side (e.g. via a flag, or via names list).
 *
 * \todo identify kernels on NMODL AST side.
 */
bool CodegenLLVMVisitor::is_kernel_function(const std::string& function_name) {
    llvm::Function* function = module->getFunction(function_name);
    if (!function)
        throw std::runtime_error("Error: function " + function_name + " does not exist\n");

    // By convention, only kernel functions have a return type of void and single argument. The
    // number of arguments check is needed to avoid LLVM void intrinsics to be considered as
    // kernels.
    if (!function->getReturnType()->isVoidTy() || !llvm::hasSingleElement(function->args()))
        return false;

    // Kernel's argument is a pointer to the instance struct type.
    llvm::Type* arg_type = function->getArg(0)->getType();
    if (auto pointer_type = llvm::dyn_cast<llvm::PointerType>(arg_type)) {
        if (pointer_type->getElementType()->isStructTy())
            return true;
    }
    return false;
}

llvm::Value* CodegenLLVMVisitor::read_from_or_write_to_instance(const ast::CodegenInstanceVar& node,
                                                                llvm::Value* maybe_value_to_store) {
    const auto& instance_name = node.get_instance_var()->get_node_name();
    const auto& member_node = node.get_member_var();
    const auto& member_name = member_node->get_node_name();

    if (!instance_var_helper.is_an_instance_variable(member_name))
        throw std::runtime_error("Error: " + member_name +
                                 " is not a member of the instance variable\n");

    // Load the instance struct by its name.
    llvm::Value* instance_ptr = ir_builder.create_load(instance_name);

    // Get the pointer to the specified member.
    int member_index = instance_var_helper.get_variable_index(member_name);
    llvm::Value* member_ptr = ir_builder.get_struct_member_ptr(instance_ptr, member_index);

    // Check if the member is scalar. Load the value or store to it straight away. Otherwise, we
    // need some extra handling.
    auto codegen_var_with_type = instance_var_helper.get_variable(member_name);
    if (!codegen_var_with_type->get_is_pointer()) {
        if (maybe_value_to_store) {
            ir_builder.create_store(member_ptr, maybe_value_to_store);
            return nullptr;
        } else {
            return ir_builder.create_load(member_ptr);
        }
    }

    // Check that the member is an indexed name indeed, and that it is indexed by a named constant
    // (e.g. "id").
    const auto& member_var_name = std::dynamic_pointer_cast<ast::VarName>(member_node);
    if (!member_var_name->get_name()->is_indexed_name())
        throw std::runtime_error("Error: " + member_name + " is not an IndexedName\n");

    const auto& member_indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(
        member_var_name->get_name());
    if (!member_indexed_name->get_length()->is_name())
        throw std::runtime_error("Error: " + member_name + " must be indexed with a variable!");

    // Get the index to the member and the id used to index it.
    llvm::Value* i64_index = get_index(*member_indexed_name);
    const std::string id = member_indexed_name->get_length()->get_node_name();

    // Load the member of the instance struct.
    llvm::Value* instance_member = ir_builder.create_load(member_ptr);

    // Create a pointer to the specified element of the struct member.
    return ir_builder.load_to_or_store_from_array(id,
                                                  i64_index,
                                                  instance_member,
                                                  maybe_value_to_store);
}

llvm::Value* CodegenLLVMVisitor::read_variable(const ast::VarName& node) {
    const auto& identifier = node.get_name();

    if (identifier->is_name()) {
        return ir_builder.create_load(node.get_node_name(),
                                      /*masked=*/ir_builder.generates_predicated_ir());
    }

    if (identifier->is_indexed_name()) {
        const auto& indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
        llvm::Value* index = get_index(*indexed_name);
        return ir_builder.create_load_from_array(node.get_node_name(), index);
    }

    if (identifier->is_codegen_instance_var()) {
        const auto& instance_var = std::dynamic_pointer_cast<ast::CodegenInstanceVar>(identifier);
        return read_from_or_write_to_instance(*instance_var);
    }

    throw std::runtime_error("Error: the type of '" + node.get_node_name() +
                             "' is not supported\n");
}

void CodegenLLVMVisitor::write_to_variable(const ast::VarName& node, llvm::Value* value) {
    const auto& identifier = node.get_name();
    if (!identifier->is_name() && !identifier->is_indexed_name() &&
        !identifier->is_codegen_instance_var()) {
        throw std::runtime_error("Error: the type of '" + node.get_node_name() +
                                 "' is not supported\n");
    }

    if (identifier->is_name()) {
        ir_builder.create_store(node.get_node_name(), value);
    }

    if (identifier->is_indexed_name()) {
        const auto& indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
        llvm::Value* index = get_index(*indexed_name);
        ir_builder.create_store_to_array(node.get_node_name(), index, value);
    }

    if (identifier->is_codegen_instance_var()) {
        const auto& instance_var = std::dynamic_pointer_cast<ast::CodegenInstanceVar>(identifier);
        read_from_or_write_to_instance(*instance_var, value);
    }
}

void CodegenLLVMVisitor::wrap_kernel_functions() {
    // First, identify all kernels.
    std::vector<std::string> kernel_names;
    find_kernel_names(kernel_names);

    for (const auto& kernel_name: kernel_names) {
        // Get the kernel function.
        auto kernel = module->getFunction(kernel_name);

        // Create a wrapper void function that takes a void pointer as a single argument.
        llvm::Type* i32_type = ir_builder.get_i32_type();
        llvm::Type* void_ptr_type = ir_builder.get_i8_ptr_type();
        llvm::Function* wrapper_func = llvm::Function::Create(
            llvm::FunctionType::get(i32_type, {void_ptr_type}, /*isVarArg=*/false),
            llvm::Function::ExternalLinkage,
            "__" + kernel_name + "_wrapper",
            *module);

        // Optionally, add debug information for the wrapper function.
        if (add_debug_information) {
            debug_builder.add_function_debug_info(wrapper_func);
        }

        ir_builder.create_block_and_set_insertion_point(wrapper_func);

        // Proceed with bitcasting the void pointer to the struct pointer type, calling the kernel
        // and adding a terminator.
        llvm::Value* bitcasted = ir_builder.create_bitcast(wrapper_func->getArg(0),
                                                           kernel->getArg(0)->getType());
        ValueVector args;
        args.push_back(bitcasted);
        ir_builder.create_function_call(kernel, args, /*use_result=*/false);

        // Create a 0 return value and a return instruction.
        ir_builder.create_i32_constant(0);
        ir_builder.create_return(ir_builder.pop_last_value());
    }
}


/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/


void CodegenLLVMVisitor::visit_binary_expression(const ast::BinaryExpression& node) {
    const auto& op = node.get_op().get_value();

    // Process rhs first, since lhs is handled differently for assignment and binary
    // operators.
    llvm::Value* rhs = accept_and_get(node.get_rhs());
    if (op == ast::BinaryOp::BOP_ASSIGN) {
        auto var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
        if (!var)
            throw std::runtime_error("Error: only 'VarName' assignment is supported\n");

        write_to_variable(*var, rhs);
        return;
    }

    llvm::Value* lhs = accept_and_get(node.get_lhs());
    ir_builder.create_binary_op(lhs, rhs, op);
}

void CodegenLLVMVisitor::visit_statement_block(const ast::StatementBlock& node) {
    const auto& statements = node.get_statements();
    for (const auto& statement: statements) {
        if (is_supported_statement(*statement))
            statement->accept(*this);
    }
}

void CodegenLLVMVisitor::visit_boolean(const ast::Boolean& node) {
    ir_builder.create_boolean_constant(node.get_value());
}

/**
 * Currently, this functions is very similar to visiting the binary operator. However, the
 * difference here is that the writes to the LHS variable must be atomic. These has a particular
 * use case in synapse kernels. For simplicity, we choose not to support atomic writes at this
 * stage and emit a warning.
 *
 * \todo support this properly.
 */
void CodegenLLVMVisitor::visit_codegen_atomic_statement(const ast::CodegenAtomicStatement& node) {
    if (vector_width > 1)
        logger->warn("Atomic operations are not supported");

    // Support only assignment for now.
    llvm::Value* rhs = accept_and_get(node.get_rhs());
    if (node.get_atomic_op().get_value() != ast::BinaryOp::BOP_ASSIGN)
        throw std::runtime_error(
            "Error: only assignment is supported for CodegenAtomicStatement\n");
    const auto& var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
    if (!var)
        throw std::runtime_error("Error: only 'VarName' assignment is supported\n");

    // Process the assignment as if it was non-atomic.
    if (vector_width > 1)
        logger->warn("Treating write as non-atomic");
    write_to_variable(*var, rhs);
}

// Generating FOR loop in LLVM IR creates the following structure:
//
//  +---------------------------+
//  | <code before for loop>    |
//  | <for loop initialisation> |
//  | br %cond                  |
//  +---------------------------+
//                |
//                V
//  +-----------------------------+
//  | <condition code>            |
//  | %cond = ...                 |<------+
//  | cond_br %cond, %body, %exit |       |
//  +-----------------------------+       |
//      |                 |               |
//      |                 V               |
//      |     +------------------------+  |
//      |     | <body code>            |  |
//      |     | br %inc                |  |
//      |     +------------------------+  |
//      |                 |               |
//      |                 V               |
//      |     +------------------------+  |
//      |     | <increment code>       |  |
//      |      | br %cond              |  |
//      |     +------------------------+  |
//      |                 |               |
//      |                 +---------------+
//      V
//  +---------------------------+
//  | <code after for loop>     |
//  +---------------------------+
void CodegenLLVMVisitor::visit_codegen_for_statement(const ast::CodegenForStatement& node) {
    // Condition and increment blocks must be scalar.
    ir_builder.generate_scalar_ir();

    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = ir_builder.get_current_block();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Create the basic blocks for FOR loop.
    llvm::BasicBlock* for_cond =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.cond", func, next);
    llvm::BasicBlock* for_body =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.body", func, next);
    llvm::BasicBlock* for_inc = llvm::BasicBlock::Create(*context, /*Name=*/"for.inc", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"for.exit", func, next);

    // First, initialize the loop in the same basic block. If processing the remainder of the loop,
    // no initialization happens.
    const auto& main_loop_initialization = node.get_initialization();
    if (main_loop_initialization)
        main_loop_initialization->accept(*this);

    // Branch to condition basic block and insert condition code there.
    ir_builder.create_br_and_set_insertion_point(for_cond);

    // Extract the condition to decide whether to branch to the loop body or loop exit.
    llvm::Value* cond = accept_and_get(node.get_condition());
    llvm::BranchInst* loop_br = ir_builder.create_cond_br(cond, for_body, exit);
    ir_builder.set_loop_metadata(loop_br);
    ir_builder.set_insertion_point(for_body);

    // If not processing remainder of the loop, start vectorization.
    if (vector_width > 1 && main_loop_initialization)
        ir_builder.generate_vector_ir();

    // Generate code for the loop body and create the basic block for the increment.
    const auto& statement_block = node.get_statement_block();
    statement_block->accept(*this);
    ir_builder.generate_scalar_ir();
    ir_builder.create_br_and_set_insertion_point(for_inc);

    // Process the increment.
    node.get_increment()->accept(*this);

    // Create a branch to condition block, then generate exit code out of the loop.
    ir_builder.create_br(for_cond);
    ir_builder.set_insertion_point(exit);
}


void CodegenLLVMVisitor::visit_codegen_function(const ast::CodegenFunction& node) {
    const auto& name = node.get_node_name();
    const auto& arguments = node.get_arguments();

    // Create the entry basic block of the function/procedure and point the local named values table
    // to the symbol table.
    llvm::Function* func = module->getFunction(name);
    ir_builder.create_block_and_set_insertion_point(func);
    ir_builder.set_function(func);

    // When processing a function, it returns a value named <function_name> in NMODL. Therefore, we
    // first run RenameVisitor to rename it into ret_<function_name>. This will aid in avoiding
    // symbolic conflicts.
    std::string return_var_name = "ret_" + name;
    const auto& block = node.get_statement_block();
    visitor::RenameVisitor v(name, return_var_name);
    block->accept(v);

    // Allocate parameters on the stack and add them to the symbol table.
    ir_builder.allocate_function_arguments(func, arguments);

    // Process function or procedure body. If the function is a compute kernel, enable
    // vectorization. If so, the return statement is handled in a separate visitor.
    if (vector_width > 1 && is_kernel_function(name)) {
        ir_builder.generate_vector_ir();
        block->accept(*this);
        ir_builder.generate_scalar_ir();
    } else {
        block->accept(*this);
    }

    // If function is a compute kernel, add a void terminator explicitly, since there is no
    // `CodegenReturnVar` node. Also, set the necessary attributes.
    if (is_kernel_function(name)) {
        ir_builder.set_kernel_attributes();
        ir_builder.create_return();
    }

    // Clear local values stack and remove the pointer to the local symbol table.
    ir_builder.clear_function();
}

void CodegenLLVMVisitor::visit_codegen_return_statement(const ast::CodegenReturnStatement& node) {
    if (!node.get_statement()->is_name())
        throw std::runtime_error("Error: CodegenReturnStatement must contain a name node\n");

    std::string ret = "ret_" + ir_builder.get_current_function_name();
    llvm::Value* ret_value = ir_builder.create_load(ret);
    ir_builder.create_return(ret_value);
}

void CodegenLLVMVisitor::visit_codegen_var_list_statement(
    const ast::CodegenVarListStatement& node) {
    llvm::Type* scalar_type = get_codegen_var_type(*node.get_var_type());
    for (const auto& variable: node.get_variables()) {
        const auto& identifier = variable->get_name();
        std::string name = variable->get_node_name();

        // Local variable can be a scalar (Node AST class) or an array (IndexedName AST class). For
        // each case, create memory allocations.
        if (identifier->is_indexed_name()) {
            const auto& indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
            int length = get_num_elements(*indexed_name);
            ir_builder.create_array_alloca(name, scalar_type, length);
        } else if (identifier->is_name()) {
            ir_builder.create_scalar_or_vector_alloca(name, scalar_type);
        } else {
            throw std::runtime_error("Error: unsupported local variable type\n");
        }
    }
}

void CodegenLLVMVisitor::visit_double(const ast::Double& node) {
    ir_builder.create_fp_constant(node.get_value());
}

void CodegenLLVMVisitor::visit_function_block(const ast::FunctionBlock& node) {
    // do nothing. \todo: remove old function blocks from ast.
}

void CodegenLLVMVisitor::visit_function_call(const ast::FunctionCall& node) {
    const auto& name = node.get_node_name();
    llvm::Function* func = module->getFunction(name);
    if (func) {
        create_function_call(func, name, node.get_arguments());
    } else {
        auto symbol = program_symtab->lookup(name);
        if (symbol && symbol->has_any_property(symtab::syminfo::NmodlType::extern_method)) {
            create_external_function_call(name, node.get_arguments());
        } else {
            throw std::runtime_error("Error: unknown function name: " + name + "\n");
        }
    }
}

void CodegenLLVMVisitor::visit_if_statement(const ast::IfStatement& node) {
    // If vectorizing the compute kernel with control flow, process it separately.
    if (vector_width > 1 && ir_builder.vectorizing()) {
        create_vectorized_control_flow_block(node);
        return;
    }

    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = ir_builder.get_current_block();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a true block and a merge block where the control flow merges.
    llvm::BasicBlock* true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    // Add condition to the current block.
    llvm::Value* cond = accept_and_get(node.get_condition());

    // Process the true block.
    ir_builder.set_insertion_point(true_block);
    node.get_statement_block()->accept(*this);
    ir_builder.create_br(merge_block);

    // Save the merge block and proceed with codegen for `else if` statements.
    llvm::BasicBlock* exit = merge_block;
    for (const auto& else_if: node.get_elseifs()) {
        // Link the current block to the true and else blocks.
        llvm::BasicBlock* else_block =
            llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.set_insertion_point(curr_block);
        ir_builder.create_cond_br(cond, true_block, else_block);

        // Process else block.
        ir_builder.set_insertion_point(else_block);
        cond = accept_and_get(else_if->get_condition());

        // Reassign true and merge blocks respectively. Note that the new merge block has to be
        // connected to the old merge block (tmp).
        true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        llvm::BasicBlock* tmp = merge_block;
        merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.set_insertion_point(merge_block);
        ir_builder.create_br(tmp);

        // Process true block.
        ir_builder.set_insertion_point(true_block);
        else_if->get_statement_block()->accept(*this);
        ir_builder.create_br(merge_block);
        curr_block = else_block;
    }

    // Finally, generate code for `else` statement if it exists.
    const auto& elses = node.get_elses();
    llvm::BasicBlock* else_block;
    if (elses) {
        else_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.set_insertion_point(else_block);
        elses->get_statement_block()->accept(*this);
        ir_builder.create_br(merge_block);
    } else {
        else_block = merge_block;
    }
    ir_builder.set_insertion_point(curr_block);
    ir_builder.create_cond_br(cond, true_block, else_block);
    ir_builder.set_insertion_point(exit);
}

void CodegenLLVMVisitor::visit_integer(const ast::Integer& node) {
    ir_builder.create_i32_constant(node.get_value());
}

void CodegenLLVMVisitor::visit_program(const ast::Program& node) {
    // Before generating LLVM:
    //   - convert function and procedure blocks into CodegenFunctions
    //   - gather information about AST. For now, information about functions
    //     and procedures is used only.
    CodegenLLVMHelperVisitor v{vector_width};
    const auto& functions = v.get_codegen_functions(node);
    instance_var_helper = v.get_instance_var_helper();
    program_symtab = node.get_symbol_table();
    std::string kernel_id = v.get_kernel_id();

    // Initialize the builder for this NMODL program.
    ir_builder.initialize(*program_symtab, kernel_id);

    // Create compile unit if adding debug information to the module.
    if (add_debug_information) {
        debug_builder.create_compile_unit(*module, module->getModuleIdentifier(), output_dir);
    }

    // For every function, generate its declaration. Thus, we can look up
    // `llvm::Function` in the symbol table in the module.
    for (const auto& func: functions) {
        create_function_declaration(*func);
    }

    // Set the AST symbol table.
    program_symtab = node.get_symbol_table();

    // Proceed with code generation. Right now, we do not do
    //     node.visit_children(*this);
    // The reason is that the node may contain AST nodes for which the visitor functions have been
    // defined. In our implementation we assume that the code generation is happening within the
    // function scope. To avoid generating code outside of functions, visit only them for now.
    // \todo: Handle what is mentioned here.
    for (const auto& func: functions) {
        visit_codegen_function(*func);
    }

    // Finalize the debug information.
    if (add_debug_information) {
        debug_builder.finalize();
    }

    // Verify the generated LLVM IR module.
    std::string error;
    llvm::raw_string_ostream ostream(error);
    if (verifyModule(*module, &ostream)) {
        throw std::runtime_error("Error: incorrect IR has been generated!\n" + ostream.str());
    }

    if (opt_level_ir) {
        logger->info("Running LLVM optimisation passes");
        utils::initialise_optimisation_passes();
        utils::optimise_module(*module, opt_level_ir);
    }

    // Optionally, replace LLVM math intrinsics with vector library calls.
    if (vector_width > 1) {
#if LLVM_VERSION_MAJOR < 13
        logger->warn(
            "This version of LLVM does not support replacement of LLVM intrinsics with vector "
            "library calls");
#else
        // First, get the target library information and add vectorizable functions for the
        // specified vector library.
        llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
        llvm::TargetLibraryInfoImpl target_lib_info = llvm::TargetLibraryInfoImpl(triple);
        add_vectorizable_functions_from_vec_lib(target_lib_info, triple);

        // Run passes that replace math intrinsics.
        llvm::legacy::FunctionPassManager fpm(module.get());
        fpm.add(new llvm::TargetLibraryInfoWrapperPass(target_lib_info));
        fpm.add(new llvm::ReplaceWithVeclibLegacy);
        fpm.doInitialization();
        for (auto& function: module->getFunctionList()) {
            if (!function.isDeclaration())
                fpm.run(function);
        }
        fpm.doFinalization();
#endif
    }

    // If the output directory is specified, save the IR to .ll file.
    if (output_dir != ".") {
        utils::save_ir_to_ll_file(*module, output_dir + "/" + mod_filename);
    }

    logger->debug("Dumping generated IR...\n" + dump_module());
    // Setup CodegenHelper for C++ wrapper file
    setup(node);
    print_wrapper_routines();
    print_target_file();
}

void CodegenLLVMVisitor::print_wrapper_headers_include() {
    print_standard_includes();
    print_coreneuron_includes();
}

void CodegenLLVMVisitor::print_mechanism_range_var_structure() {
    printer->add_newline(2);
    printer->add_line("/** Instance Struct passed as argument to LLVM IR kernels */");
    printer->start_block("struct {} "_format(mod_filename + instance_struct_type_name));
    for (const auto& variable: instance_var_helper.instance->get_codegen_vars()) {
        auto is_pointer = variable->get_is_pointer();
        auto name = to_nmodl(variable->get_name());
        auto qualifier = is_constant_variable(name) ? k_const() : "";
        auto nmodl_type = variable->get_type()->get_type();
        auto pointer = is_pointer ? "*" : "";
        auto var_name = variable->get_node_name();
        switch (nmodl_type) {
#define DISPATCH(type, c_type)                                                              \
    case type:                                                                              \
        printer->add_line("{}{}{} {}{};"_format(                                            \
            qualifier, c_type, pointer, is_pointer ? ptr_type_qualifier() : "", var_name)); \
        break;

            DISPATCH(ast::AstNodeType::DOUBLE, "double");
            DISPATCH(ast::AstNodeType::INTEGER, "int");

#undef DISPATCH
        default:
            throw std::runtime_error("Error: unsupported type found in instance struct");
        }
    }
    printer->end_block();
}

void CodegenLLVMVisitor::print_instance_variable_setup() {
    if (range_variable_setup_required()) {
        print_setup_range_variable();
    }

    if (shadow_vector_setup_required()) {
        print_shadow_vector_setup();
    }
    printer->add_newline(2);
    printer->add_line("/** initialize mechanism instance variables */");
    printer->start_block("static inline void setup_instance(NrnThread* nt, Memb_list* ml) ");
    printer->add_line("{0}* inst = ({0}*) mem_alloc(1, sizeof({0}));"_format(
        mod_filename + instance_struct_type_name));
    if (channel_task_dependency_enabled() && !info.codegen_shadow_variables.empty()) {
        printer->add_line("setup_shadow_vectors(inst, ml);");
    }

    std::string stride;
    printer->add_line("int pnodecount = ml->_nodecount_padded;");
    stride = "*pnodecount";

    printer->add_line("Datum* indexes = ml->pdata;");

    std::string float_type = default_float_data_type();
    std::string int_type = default_int_data_type();
    std::string float_type_pointer = float_type + "*";
    std::string int_type_pointer = int_type + "*";

    int id = 0;
    std::vector<std::string> variables_to_free;

    for (auto& var: info.codegen_float_variables) {
        auto name = var->get_name();
        auto range_var_type = get_range_var_float_type(var);
        if (float_type == range_var_type) {
            auto variable = "ml->data+{}{}"_format(id, stride);
            auto device_variable = get_variable_device_pointer(variable, float_type_pointer);
            printer->add_line("inst->{} = {};"_format(name, device_variable));
        } else {
            printer->add_line("inst->{} = setup_range_variable(ml->data+{}{}, pnodecount);"_format(
                name, id, stride));
            variables_to_free.push_back(name);
        }
        id += var->get_length();
    }

    for (auto& var: info.codegen_int_variables) {
        auto name = var.symbol->get_name();
        std::string variable = name;
        std::string type = "";
        if (var.is_index || var.is_integer) {
            variable = "ml->pdata";
            type = int_type_pointer;
        } else if (var.is_vdata) {
            variable = "nt->_vdata";
            type = "void**";
        } else {
            variable = "nt->_data";
            type = info.artificial_cell ? "void*" : float_type_pointer;
        }
        auto device_variable = get_variable_device_pointer(variable, type);
        printer->add_line("inst->{} = {};"_format(name, device_variable));
    }

    int index_id = 0;
    // for integer variables, there should be index
    for (const auto& int_var: info.codegen_int_variables) {
        std::string var_name = int_var.symbol->get_name() + "_index";
        // Create for loop that instantiates the ion_<var>_index with
        // indexes[<var_id>*pdnodecount+id] where id is from 0 to nodecount
        printer->add_line("inst->{} = mem_alloc(pnodecount, sizeof(int));"_format(var_name));
        print_channel_iteration_loop("0", "pnodecount");
        printer->add_line("inst->{}[id] = indexes[{}*pnodecount+id];"_format(var_name, index_id));
        printer->end_block(1);
        index_id++;
    }

    // Pass voltage pointer to the the instance struct
    printer->add_line("inst->voltage = nt->_actual_v;");

    // Pass ml->nodeindices pointer to node_index
    printer->add_line("inst->node_index = ml->nodeindices;");

    // Setup global variables
    printer->add_line("inst->{0} = nt->{0};"_format(naming::NTHREAD_T_VARIABLE));
    printer->add_line("inst->{0} = nt->{0};"_format(naming::NTHREAD_DT_VARIABLE));
    printer->add_line("inst->{0} = {0};"_format(naming::CELSIUS_VARIABLE));
    printer->add_line("inst->{0} = {0};"_format(naming::SECOND_ORDER_VARIABLE));
    printer->add_line("inst->{} = ml->nodecount;"_format(naming::MECH_NODECOUNT_VAR));

    printer->add_line("ml->instance = (void*) inst;");
    printer->end_block(3);

    printer->add_line("/** cleanup mechanism instance variables */");
    printer->start_block("static inline void cleanup_instance(Memb_list* ml) ");
    printer->add_line(
        "{0}* inst = ({0}*) ml->instance;"_format(mod_filename + instance_struct_type_name));
    for (const auto& int_var: info.codegen_int_variables) {
        std::string var_name = int_var.symbol->get_name() + "_index";
        printer->add_line("mem_free((void*)inst->{});"_format(var_name));
    }
    if (range_variable_setup_required()) {
        for (auto& var: variables_to_free) {
            printer->add_line("mem_free((void*)inst->{});"_format(var));
        }
    }
    printer->add_line("mem_free((void*)inst);");
    printer->end_block(1);
}

CodegenLLVMVisitor::ParamVector CodegenLLVMVisitor::get_compute_function_parameter() {
    auto params = ParamVector();
    params.emplace_back(param_type_qualifier(),
                        "{}*"_format(mod_filename + instance_struct_type_name),
                        ptr_type_qualifier(),
                        "inst");
    return params;
}

void CodegenLLVMVisitor::print_backend_compute_routine_decl() {
    auto params = get_compute_function_parameter();
    auto compute_function = compute_method_name(BlockType::Initial);

    printer->add_newline(2);
    printer->add_line("extern void {}({});"_format(compute_function, get_parameter_str(params)));

    if (info.nrn_cur_required()) {
        compute_function = compute_method_name(BlockType::Equation);
        printer->add_line(
            "extern void {}({});"_format(compute_function, get_parameter_str(params)));
    }

    if (info.nrn_state_required()) {
        compute_function = compute_method_name(BlockType::State);
        printer->add_line(
            "extern void {}({});"_format(compute_function, get_parameter_str(params)));
    }
}

// Copied from CodegenIspcVisitor
void CodegenLLVMVisitor::print_wrapper_routine(const std::string& wrapper_function,
                                               BlockType type) {
    static const auto args = "NrnThread* nt, Memb_list* ml, int type";
    const auto function_name = method_name(wrapper_function);
    auto compute_function = compute_method_name(type);

    printer->add_newline(2);
    printer->start_block("void {}({})"_format(function_name, args));
    printer->add_line("int nodecount = ml->nodecount;");
    // clang-format off
    printer->add_line("{0}* {1}inst = ({0}*) ml->instance;"_format(mod_filename +
        instance_struct_type_name, ptr_type_qualifier()));
    // clang-format on

    if (type == BlockType::Initial) {
        printer->add_newline();
        printer->add_line("setup_instance(nt, ml);");
        printer->add_newline();
        printer->start_block("if (_nrn_skip_initmodel)");
        printer->add_line("return;");
        printer->end_block();
        printer->add_newline();
    }

    printer->add_line("{}(inst);"_format(compute_function));
    printer->end_block();
    printer->add_newline();
}

void CodegenLLVMVisitor::print_backend_compute_routine() {
    print_wrapper_routine(naming::NRN_INIT_METHOD, BlockType::Initial);
    print_wrapper_routine(naming::NRN_CUR_METHOD, BlockType::Equation);
    print_wrapper_routine(naming::NRN_STATE_METHOD, BlockType::State);
}

void CodegenLLVMVisitor::print_data_structures() {
    print_mechanism_global_var_structure();
    print_mechanism_range_var_structure();
    print_ion_var_structure();
}

void CodegenLLVMVisitor::print_compute_functions() {
    print_top_verbatim_blocks();
    print_backend_compute_routine_decl();
    print_net_send_buffering();
    print_net_init();
    print_watch_activate();
    print_watch_check();
    print_net_receive_kernel();
    print_net_receive();
    print_net_receive_buffering();
    print_backend_compute_routine();
}

void CodegenLLVMVisitor::print_wrapper_routines() {
    printer = wrapper_printer;
    wrapper_codegen = true;
    print_backend_info();
    print_wrapper_headers_include();
    print_namespace_begin();

    CodegenCVisitor::print_nmodl_constants();
    print_mechanism_info();
    print_data_structures();
    print_global_variables_for_hoc();
    print_common_getters();

    print_memory_allocation_routine();
    print_thread_memory_callbacks();
    print_abort_routine();
    print_global_variable_setup();
    print_instance_variable_setup();
    print_nrn_alloc();
    print_compute_functions();
    print_check_table_thread_function();
    print_mechanism_register();
    print_namespace_end();
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    // do nothing. \todo: remove old procedures from ast.
}

void CodegenLLVMVisitor::visit_unary_expression(const ast::UnaryExpression& node) {
    ast::UnaryOp op = node.get_op().get_value();
    llvm::Value* value = accept_and_get(node.get_expression());
    ir_builder.create_unary_op(value, op);
}

void CodegenLLVMVisitor::visit_var_name(const ast::VarName& node) {
    llvm::Value* value = read_variable(node);
    ir_builder.maybe_replicate_value(value);
}

void CodegenLLVMVisitor::visit_while_statement(const ast::WhileStatement& node) {
    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = ir_builder.get_current_block();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a header and the body blocks.
    llvm::BasicBlock* header = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    ir_builder.create_br_and_set_insertion_point(header);


    // Generate code for condition and create branch to the body block.
    llvm::Value* condition = accept_and_get(node.get_condition());
    ir_builder.create_cond_br(condition, body, exit);

    ir_builder.set_insertion_point(body);
    node.get_statement_block()->accept(*this);
    ir_builder.create_br(header);

    ir_builder.set_insertion_point(exit);
}

}  // namespace codegen
}  // namespace nmodl

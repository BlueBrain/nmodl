#include <algorithm>

#include "ast/codegen_var_type.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

#include "codegen_data_helper.hpp"

namespace nmodl {
namespace codegen {

// scalar variables with default values
const double default_nthread_dt_value = 0.025;
const double default_nthread_t_value = 100.0;
const double default_celsius_value = 34.0;
const int default_second_order_value = 0;

// cleanup all members and struct base pointer
CodegenInstanceData::~CodegenInstanceData() {
    // first free num_ptr_members members which are pointers
    for (size_t i = 0; i < num_ptr_members; i++) {
        free(members[i]);
    }
    // and then pointer to container struct
    free(base_ptr);
}

/**
 * \todo : various things can be improved here
 * - if variable is voltage then initialization range could be -65 to +65
 * - if variable is double or float then those could be initialize with
 *   "some" floating point value between range like 1.0 to 100.0. Note
 *   it would be nice to have unique values to avoid errors like division
 *   by zero. We have simple implementation that is taking care of this.
 * - if variable is integer then initialization range must be between
 *   0 and num_elements. In practice, num_elements is number of instances
 *   of a particular mechanism. This would be <= number of compartments
 *   in the cell. For now, just initialize integer variables from 0 to
 *   num_elements - 1.
 */
void initialize_variable(const std::shared_ptr<ast::CodegenVarWithType>& var,
                         void* ptr,
                         size_t initial_value,
                         size_t num_elements) {
    ast::AstNodeType type = var->get_type()->get_type();
    const std::string& name = var->get_name()->get_node_name();

    if (type == ast::AstNodeType::DOUBLE) {
        const auto& generated_double_data = generate_dummy_data<double>(initial_value,
                                                                        num_elements);
        double* data = (double*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_double_data[i];
        }
    } else if (type == ast::AstNodeType::FLOAT) {
        const auto& generated_float_data = generate_dummy_data<float>(initial_value, num_elements);
        float* data = (float*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_float_data[i];
        }
    } else if (type == ast::AstNodeType::INTEGER) {
        const auto& generated_int_data = generate_dummy_data<int>(initial_value, num_elements);
        int* data = (int*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_int_data[i];
        }
    } else {
        throw std::runtime_error("Unhandled data type during initialize_variable");
    };
}

CodegenInstanceData CodegenDataHelper::create_data(size_t num_elements, size_t seed) {
    // alignment with 64-byte to generate aligned loads/stores
    const unsigned NBYTE_ALIGNMENT = 64;

    // get variable information
    const auto& variables = instance->get_codegen_vars();

    // start building data
    CodegenInstanceData data;
    data.num_elements = num_elements;

    // base pointer to instance object
    void* base = nullptr;

    // max size of each member : pointer / double has maximum size
    size_t member_size = std::max(sizeof(double), sizeof(double*));

    // allocate instance object with memory alignment
    posix_memalign(&base, NBYTE_ALIGNMENT, member_size * variables.size());
    data.base_ptr = base;
    data.num_bytes += member_size * variables.size();

    size_t offset = 0;
    void* ptr = base;
    size_t variable_index = 0;

    // allocate each variable and allocate memory at particular offset in base pointer
    for (auto& var: variables) {
        // only process until first non-pointer variable
        if (!var->get_is_pointer()) {
            break;
        }

        // check type of variable and it's size
        size_t member_size = 0;
        ast::AstNodeType type = var->get_type()->get_type();
        if (type == ast::AstNodeType::DOUBLE) {
            member_size = sizeof(double);
        } else if (type == ast::AstNodeType::FLOAT) {
            member_size = sizeof(float);
        } else if (type == ast::AstNodeType::INTEGER) {
            member_size = sizeof(int);
        }

        // allocate memory and setup a pointer
        void* member;
        posix_memalign(&member, NBYTE_ALIGNMENT, member_size * num_elements);

        // integer values are often offsets so they must start from
        // 0 to num_elements-1 to avoid out of bound accesses.
        int initial_value = variable_index;
        if (type == ast::AstNodeType::INTEGER) {
            initial_value = 0;
        }
        initialize_variable(var, member, initial_value, num_elements);
        data.num_bytes += member_size * num_elements;

        // copy address at specific location in the struct
        memcpy(ptr, &member, sizeof(double*));

        data.offsets.push_back(offset);
        data.members.push_back(member);
        data.num_ptr_members++;

        // all pointer types are of same size, so just use double*
        offset += sizeof(double*);
        ptr = (char*) base + offset;

        variable_index++;
    }

    // we are now switching from pointer type to next member type (e.g. double)
    // ideally we should use padding but switching from double* to double should
    // already meet alignment requirements
    for (auto& var: variables) {
        // process only scalar elements
        if (var->get_is_pointer()) {
            continue;
        }
        ast::AstNodeType type = var->get_type()->get_type();
        const std::string& name = var->get_name()->get_node_name();

        // some default values for standard parameters
        double value = 0;
        if (name == naming::NTHREAD_DT_VARIABLE) {
            value = default_nthread_dt_value;
        } else if (name == naming::NTHREAD_T_VARIABLE) {
            value = default_nthread_t_value;
        } else if (name == naming::CELSIUS_VARIABLE) {
            value = default_celsius_value;
        } else if (name == CodegenLLVMHelperVisitor::NODECOUNT_VAR) {
            value = num_elements;
        } else if (name == naming::SECOND_ORDER_VARIABLE) {
            value = default_second_order_value;
        }

        if (type == ast::AstNodeType::DOUBLE) {
            *((double*) ptr) = value;
            data.offsets.push_back(offset);
            data.members.push_back(ptr);
            offset += sizeof(double);
            ptr = (char*) base + offset;
        } else if (type == ast::AstNodeType::FLOAT) {
            *((float*) ptr) = float(value);
            data.offsets.push_back(offset);
            data.members.push_back(ptr);
            offset += sizeof(float);
            ptr = (char*) base + offset;
        } else if (type == ast::AstNodeType::INTEGER) {
            *((int*) ptr) = int(value);
            data.offsets.push_back(offset);
            data.members.push_back(ptr);
            offset += sizeof(int);
            ptr = (char*) base + offset;
        } else {
            throw std::runtime_error(
                "Unhandled type while allocating data in CodegenDataHelper::create_data()");
        }
    }

    return data;
}

}  // namespace codegen
}  // namespace nmodl

#include <algorithm>

#include "ast/codegen_var_type.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

#include "codegen_data_helper.hpp"

namespace nmodl {
namespace codegen {

/**
 * \class PaddingHelper
 * \brief Helper to calculate padding/alignment at runtime
 *
 * C/C++ has different padding/alignment requirements based
 * on target platform. This simple struct help to calculate
 * alignment requirements at runtime.
 * See https://en.wikipedia.org/wiki/Data_structure_alignment.
 *
 * TODO : We are not using this as we rely on inserting variabels in
 * order into instance struct : pointer, double, float, int
 */
struct PaddingHelper {
    // to calculate alignment/padding requirement we
    // make pair of different variables with respective
    // types to find alignment requirement

    // pointer alignment requirement
    char a;
    double* b;

    // double alignment requirement
    char c;
    double d;

    // float alignment requirement
    char e;
    float f;

    // integer alignment requirement
    char g;
    int h;

    // char is always 1 byte. We subtract base address of char
    // from "next member" because padding inserted will be equal
    // to alignment requirement of the "next member".

    size_t pointer_alignment() {
        return (char*) (&b) - &a;
    }

    size_t double_alignment() {
        return (char*) (&d) - &c;
    }

    size_t float_alignment() {
        return (char*) (&f) - &e;
    }

    size_t int_alignment() {
        return (char*) (&h) - &g;
    }
};

std::vector<double> generate_double_data(const size_t& initial_value, const size_t& num_elements) {
    std::vector<double> data(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        data[i] = initial_value + i*1e-15;
    }

    return data;
}

std::vector<float> generate_float_data(const size_t& initial_value, const size_t& num_elements) {
    std::vector<float> data(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        data[i] = initial_value + i*1e-6;
    }

    return data;
}

std::vector<int> generate_int_data(const size_t& initial_value, const size_t& num_elements) {
    std::vector<int> data(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        data[i] = initial_value + i;
    }

    return data;
}

void initialize_variable(const std::shared_ptr<ast::CodegenVarWithType>& var,
                         void* ptr,
                         size_t initial_value,
                         size_t num_elements) {
    ast::AstNodeType type = var->get_type()->get_type();
    const std::string& name = var->get_name()->get_node_name();

    // todos : various things one need to take care of here
    //   - if variable is voltage then initialization range could be -65 to +65
    //   - if variable is double or float then those could be initialize with
    //     "some" floating point value between range like 1.0 to 100.0. Note
    //     it would be nice to have unique values to avoid errors like division
    //     by zero
    //   - if variable is integer then initialization range must be between
    //     0 and num_elements. In practice, num_elements is number of instances
    //     of a particular mechanism. This would be <= number of compartments
    //     in the cell. For now, just initialize integer variables from 0 to
    //     num_elements - 1.

    if (type == ast::AstNodeType::DOUBLE) {
        std::vector<double> generated_double_data = generate_double_data(initial_value,
                                                                         num_elements);
        double* data = (double*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_double_data[i];
        }
    } else if (type == ast::AstNodeType::FLOAT) {
        std::vector<float> generated_float_data = generate_float_data(initial_value, num_elements);
        float* data = (float*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_float_data[i];
        }
    } else if (type == ast::AstNodeType::INTEGER) {
        std::vector<int> generated_int_data = generate_int_data(initial_value, num_elements);
        int* data = (int*) ptr;
        for (size_t i = 0; i < num_elements; i++) {
            data[i] = generated_int_data[i];
        }
    } else {
        throw std::runtime_error("Unhandled data type during initialize_variable");
    };
}

CodegenInstanceData CodegenDataHelper::create_data(size_t num_elements, size_t seed) {
    const unsigned NBYTE_ALIGNMENT = 64;

    CodegenInstanceData data;
    data.num_elements = num_elements;

    const auto& variables = instance->get_codegen_vars();

    // base pointer to instance object
    void* base = nullptr;
    // max size of each member : pointer / double has maximum size
    size_t member_size = std::max(sizeof(double), sizeof(double*));
    // allocate instance object with memory alignment
    posix_memalign(&base, NBYTE_ALIGNMENT, member_size * variables.size());

    size_t offset = 0;
    void* ptr = base;
    size_t variable_index = 0;
    for (auto& var: variables) {
        // only process until first non-pointer variable
        if (!var->get_is_pointer()) {
            break;
        }

        unsigned member_size = 0;
        ast::AstNodeType type = var->get_type()->get_type();
        if (type == ast::AstNodeType::DOUBLE) {
            member_size = sizeof(double);
        } else if (type == ast::AstNodeType::FLOAT) {
            member_size = sizeof(float);
        } else if (type == ast::AstNodeType::INTEGER) {
            member_size = sizeof(int);
        }

        posix_memalign(&ptr, NBYTE_ALIGNMENT, member_size * num_elements);
        initialize_variable(var, ptr, variable_index, num_elements);
        data.offsets.push_back(offset);
        data.members.push_back(ptr);

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
            value = 0.025;
        } else if (name == naming::NTHREAD_T_VARIABLE) {
            value = 100.0;
        } else if (name == naming::CELSIUS_VARIABLE) {
            value = 34.0;
        } else if (name == CodegenLLVMHelperVisitor::NODECOUNT_VAR) {
            value = num_elements;
        } else if (name == naming::SECOND_ORDER_VARIABLE) {
            value = 0;
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
#ifndef NMODL_CODEGEN_C_OMP_VISITOR_HPP
#define NMODL_CODEGEN_C_OMP_VISITOR_HPP

#include "codegen/c/codegen_c_visitor.hpp"


/**
 * \class CodegenCOmpVisitor
 * \brief Visitor for printing c code with OpenMP backend
 *
 * \todo :
 *      - handle define i.e. macro statement printing
 *      - return statement in the verbatim block of inline function not handled (e.g. netstim.mod)
 */
class CodegenCOmpVisitor : public CodegenCVisitor {
  protected:
    /// name of the code generation backend
    std::string backend_name() override;


    /// common includes : standard c/c++, coreneuron and backend specific
    void print_backend_includes() override;


    /// channel execution with dependency (backend specific)
    bool channel_task_dependency_enabled() override;


    /// channel iterations from which task can be created
    void print_channel_iteration_task_begin(BlockType type) override;


    /// end of task for channel iteration
    void print_channel_iteration_task_end() override;


    /// backend specific block start for tiling on channel iteration
    void print_channel_iteration_tiling_block_begin(BlockType type) override;


    /// backend specific block end for tiling on channel iteration
    void print_channel_iteration_tiling_block_end() override;


    /// ivdep like annotation for channel iterations
    void print_channel_iteration_block_parallel_hint() override;


    /// atomic update pragma for reduction statements
    void print_atomic_reduction_pragma() override;


    /// use of shadow updates at channel level required
    bool block_require_shadow_update(BlockType type) override;


  public:
    CodegenCOmpVisitor(std::string mod_file,
                       std::string output_dir,
                       bool aos,
                       std::string float_type)
        : CodegenCVisitor(mod_file, output_dir, aos, float_type) {
    }

    CodegenCOmpVisitor(std::string mod_file,
                       std::stringstream& stream,
                       bool aos,
                       std::string float_type)
        : CodegenCVisitor(mod_file, stream, aos, float_type) {
    }
};


#endif

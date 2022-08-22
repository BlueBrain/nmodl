/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <memory>
#include <set>

#include <pybind11/pybind11.h>

#include "ast/program.hpp"
#include "codegen/codegen_driver.hpp"
#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "pybind/pybind_utils.hpp"
#include "visitors/visitor_utils.hpp"

#ifdef NMODL_LLVM_BACKEND
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "test/benchmark/llvm_benchmark.hpp"
#endif

/**
 * \dir
 * \brief Python Interface Implementation
 *
 * \file
 * \brief Top level nmodl Python module implementation
 */


namespace py = pybind11;
using namespace pybind11::literals;


namespace nmodl {

/** \brief docstring of Python exposed API */
namespace docstring {

static const char* driver = R"(
    This is the NmodlDriver class documentation
)";

static const char* driver_ast = R"(
    Get ast

    Returns:
        Instance of :py:class:`Program`
)";

static const char* driver_parse_string = R"(
    Parse NMODL provided as a string

    Args:
        input (str): C code as string
    Returns:
        AST: ast root node if success, throws an exception otherwise

    >>> ast = driver.parse_string("DEFINE NSTEP 6")
)";

static const char* driver_parse_file = R"(
    Parse NMODL provided as a file

    Args:
        filename (str): name of the C file

    Returns:
        AST: ast root node if success, throws an exception otherwise
)";

static const char* driver_parse_stream = R"(
    Parse NMODL file provided as istream

    Args:
        in (file): ifstream object

    Returns:
        AST: ast root node if success, throws an exception otherwise
)";

static const char* to_nmodl = R"(
    Given AST node, return the NMODL string representation

    Args:
        node (AST): AST node
        excludeTypes (set of AstNodeType): Excluded node types

    Returns:
        str: NMODL string representation

    >>> ast = driver.parse_string("NEURON{}")
    >>> nmodl.to_nmodl(ast)
    'NEURON {\n}\n'
)";

static const char* to_json = R"(
    Given AST node, return the JSON string representation

    Args:
        node (AST): AST node
        compact (bool): Compact node
        expand (bool): Expand node

    Returns:
        str: JSON string representation

    >>> ast = driver.parse_string("NEURON{}")
    >>> nmodl.to_json(ast, True)
    '{"Program":[{"NeuronBlock":[{"StatementBlock":[]}]}]}'
)";

#ifdef NMODL_LLVM_BACKEND
static const char* jit = R"(
    This is the Jit class documentation
)";
#endif

}  // namespace docstring


/**
 * \class PyNmodlDriver
 * \brief Class to bridge C++ NmodlDriver with Python world using pybind11
 */
class PyNmodlDriver: public nmodl::parser::NmodlDriver {
  public:
    std::shared_ptr<nmodl::ast::Program> parse_stream(py::object object) {
        py::object tiob = py::module::import("io").attr("TextIOBase");
        if (py::isinstance(object, tiob)) {
            py::detail::pythonibuf<py::str> buf(object);
            std::istream istr(&buf);
            return NmodlDriver::parse_stream(istr);
        } else {
            py::detail::pythonibuf<py::bytes> buf(object);
            std::istream istr(&buf);
            return NmodlDriver::parse_stream(istr);
        }
    }
};

#ifdef NMODL_LLVM_BACKEND
class JitDriver {
  private:
    nmodl::codegen::Platform platform;

    nmodl::codegen::CodeGenConfig cfg;
    nmodl::codegen::CodegenDriver cg_driver;

    void init_platform() {
        // Create platform abstraction.
        nmodl::codegen::PlatformID pid = cfg.llvm_gpu_name == "default"
                                             ? nmodl::codegen::PlatformID::CPU
                                             : nmodl::codegen::PlatformID::GPU;
        const std::string name = cfg.llvm_gpu_name == "default" ? cfg.llvm_cpu_name
                                                                : cfg.llvm_gpu_name;
        platform = nmodl::codegen::Platform(pid,
                                            name,
                                            cfg.llvm_gpu_target_architecture,
                                            cfg.llvm_math_library,
                                            cfg.llvm_float_type,
                                            cfg.llvm_vector_width);
        if (platform.is_gpu() && !platform.is_CUDA_gpu()) {
            throw std::runtime_error("Benchmarking is only supported on CUDA GPUs at the moment");
        }
#ifndef NMODL_LLVM_CUDA_BACKEND
        if (platform.is_CUDA_gpu()) {
            throw std::runtime_error(
                "GPU benchmarking is not supported if NMODL is not built with CUDA "
                "backend enabled.");
        }
#endif
    }

  public:
    JitDriver()
        : cg_driver(cfg) {
        init_platform();
    }

    explicit JitDriver(const nmodl::codegen::CodeGenConfig& cfg)
        : cfg(cfg)
        , cg_driver(cfg) {
        init_platform();
    }


    benchmark::BenchmarkResults run(std::shared_ptr<nmodl::ast::Program> node,
                                    std::string& modname,
                                    int num_experiments,
                                    int instance_size,
                                    int cuda_grid_dim_x,
                                    int cuda_block_dim_x) {
        // New directory is needed to be created otherwise the directory cannot be created
        // automatically through python
        if (cfg.nmodl_ast || cfg.json_ast || cfg.json_perfstat) {
            utils::make_path(cfg.scratch_dir);
        }
        cg_driver.prepare_mod(node, modname);
        nmodl::codegen::CodegenLLVMVisitor visitor(modname, cfg.output_dir, platform, 0, false, {}, true);
        visitor.visit_program(*node);
        const GPUExecutionParameters gpu_execution_parameters{cuda_grid_dim_x, cuda_block_dim_x};
        nmodl::benchmark::LLVMBenchmark benchmark(visitor,
                                                  modname,
                                                  cfg.output_dir,
                                                  cfg.shared_lib_paths,
                                                  num_experiments,
                                                  instance_size,
                                                  platform,
                                                  cfg.llvm_opt_level_ir,
                                                  cfg.llvm_opt_level_codegen,
                                                  gpu_execution_parameters);
        return benchmark.run();
    }
};
#endif

}  // namespace nmodl

// forward declaration of submodule init functions
void init_visitor_module(py::module& m);
void init_ast_module(py::module& m);
void init_symtab_module(py::module& m);


PYBIND11_MODULE(_nmodl, m_nmodl) {
    m_nmodl.doc() = "NMODL : Source-to-Source Code Generation Framework";
    m_nmodl.attr("__version__") = nmodl::Version::NMODL_VERSION;

    py::class_<nmodl::PyNmodlDriver> nmodl_driver(m_nmodl, "NmodlDriver", nmodl::docstring::driver);
    nmodl_driver.def(py::init<>())
        .def("parse_string",
             &nmodl::PyNmodlDriver::parse_string,
             "input"_a,
             nmodl::docstring::driver_parse_string)
        .def(
            "parse_file",
            [](nmodl::PyNmodlDriver& driver, const std::string& file) {
                return driver.parse_file(file, nullptr);
            },
            "filename"_a,
            nmodl::docstring::driver_parse_file)
        .def("parse_stream",
             &nmodl::PyNmodlDriver::parse_stream,
             "in"_a,
             nmodl::docstring::driver_parse_stream)
        .def("get_ast", &nmodl::PyNmodlDriver::get_ast, nmodl::docstring::driver_ast);

    py::class_<nmodl::codegen::CodeGenConfig> cfg(m_nmodl, "CodeGenConfig");
    cfg.def(py::init([]() {
           auto cfg = std::make_unique<nmodl::codegen::CodeGenConfig>();
#ifdef NMODL_LLVM_BACKEND
           // set to more sensible defaults for python binding
           cfg->llvm_ir = true;
#endif
           return cfg;
       }))
        .def_readwrite("sympy_analytic", &nmodl::codegen::CodeGenConfig::sympy_analytic)
        .def_readwrite("sympy_pade", &nmodl::codegen::CodeGenConfig::sympy_pade)
        .def_readwrite("sympy_cse", &nmodl::codegen::CodeGenConfig::sympy_cse)
        .def_readwrite("sympy_conductance", &nmodl::codegen::CodeGenConfig::sympy_conductance)
        .def_readwrite("nmodl_inline", &nmodl::codegen::CodeGenConfig::nmodl_inline)
        .def_readwrite("nmodl_unroll", &nmodl::codegen::CodeGenConfig::nmodl_unroll)
        .def_readwrite("nmodl_const_folding", &nmodl::codegen::CodeGenConfig::nmodl_const_folding)
        .def_readwrite("nmodl_localize", &nmodl::codegen::CodeGenConfig::nmodl_localize)
        .def_readwrite("nmodl_global_to_range",
                       &nmodl::codegen::CodeGenConfig::nmodl_global_to_range)
        .def_readwrite("nmodl_local_to_range", &nmodl::codegen::CodeGenConfig::nmodl_local_to_range)
        .def_readwrite("localize_verbatim", &nmodl::codegen::CodeGenConfig::localize_verbatim)
        .def_readwrite("local_rename", &nmodl::codegen::CodeGenConfig::local_rename)
        .def_readwrite("verbatim_inline", &nmodl::codegen::CodeGenConfig::verbatim_inline)
        .def_readwrite("verbatim_rename", &nmodl::codegen::CodeGenConfig::verbatim_rename)
        .def_readwrite("force_codegen", &nmodl::codegen::CodeGenConfig::force_codegen)
        .def_readwrite("only_check_compatibility",
                       &nmodl::codegen::CodeGenConfig::only_check_compatibility)
        .def_readwrite("optimize_ionvar_copies_codegen",
                       &nmodl::codegen::CodeGenConfig::optimize_ionvar_copies_codegen)
        .def_readwrite("output_dir", &nmodl::codegen::CodeGenConfig::output_dir)
        .def_readwrite("scratch_dir", &nmodl::codegen::CodeGenConfig::scratch_dir)
        .def_readwrite("data_type", &nmodl::codegen::CodeGenConfig::data_type)
        .def_readwrite("nmodl_ast", &nmodl::codegen::CodeGenConfig::nmodl_ast)
        .def_readwrite("json_ast", &nmodl::codegen::CodeGenConfig::json_ast)
        .def_readwrite("json_perfstat", &nmodl::codegen::CodeGenConfig::json_perfstat)
#ifdef NMODL_LLVM_BACKEND
        .def_readwrite("llvm_ir", &nmodl::codegen::CodeGenConfig::llvm_ir)
        .def_readwrite("llvm_float_type", &nmodl::codegen::CodeGenConfig::llvm_float_type)
        .def_readwrite("llvm_opt_level_ir", &nmodl::codegen::CodeGenConfig::llvm_opt_level_ir)
        .def_readwrite("llvm_math_library", &nmodl::codegen::CodeGenConfig::llvm_math_library)
        .def_readwrite("llvm_no_debug", &nmodl::codegen::CodeGenConfig::llvm_no_debug)
        .def_readwrite("llvm_fast_math_flags", &nmodl::codegen::CodeGenConfig::llvm_fast_math_flags)
        .def_readwrite("llvm_cpu_name", &nmodl::codegen::CodeGenConfig::llvm_cpu_name)
        .def_readwrite("llvm_gpu_name", &nmodl::codegen::CodeGenConfig::llvm_gpu_name)
        .def_readwrite("llvm_gpu_target_architecture",
                       &nmodl::codegen::CodeGenConfig::llvm_gpu_target_architecture)
        .def_readwrite("llvm_vector_width", &nmodl::codegen::CodeGenConfig::llvm_vector_width)
        .def_readwrite("llvm_opt_level_codegen",
                       &nmodl::codegen::CodeGenConfig::llvm_opt_level_codegen)
        .def_readwrite("shared_lib_paths", &nmodl::codegen::CodeGenConfig::shared_lib_paths);

    py::class_<nmodl::JitDriver> jit_driver(m_nmodl, "Jit", nmodl::docstring::jit);
    jit_driver.def(py::init<>())
        .def(py::init<const nmodl::codegen::CodeGenConfig&>())
        .def("run",
             &nmodl::JitDriver::run,
             "node"_a,
             "modname"_a,
             "num_experiments"_a,
             "instance_size"_a,
             "cuda_grid_dim_x"_a = 1,
             "cuda_block_dim_x"_a = 1);
#else
        ;
#endif

    m_nmodl.def("to_nmodl",
                static_cast<std::string (*)(const nmodl::ast::Ast&,
                                            const std::set<nmodl::ast::AstNodeType>&)>(
                    nmodl::to_nmodl),
                "node"_a,
                "exclude_types"_a = std::set<nmodl::ast::AstNodeType>(),
                nmodl::docstring::to_nmodl);
    m_nmodl.def("to_json",
                nmodl::to_json,
                "node"_a,
                "compact"_a = false,
                "expand"_a = false,
                "add_nmodl"_a = false,
                nmodl::docstring::to_json);

    init_visitor_module(m_nmodl);
    init_ast_module(m_nmodl);
    init_symtab_module(m_nmodl);
}

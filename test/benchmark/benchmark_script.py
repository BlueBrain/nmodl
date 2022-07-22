import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess

import nmodl.dsl as nmodl

@dataclass
class CompilersConfig:
    svml_lib: str = ""
    intel_exe: str = ""
    sleef_lib: str = ""
    clang_exe: str = ""
    llc_exe: str = ""
    gcc_exe: str = ""
    nvhpc_exe: str = ""
    libdevice_lib: str = ""
    nmodl_exe: str = ""

    def get_compiler_cmd(self, compiler):
        if compiler == "intel":
            return self.intel_exe
        elif compiler == "clang":
            return self.clang_exe
        elif compiler == "gcc":
            return self.gcc_exe
        elif compiler == "nvhpc":
            return self.nvhpc_exe
        else:
            raise Exception("Unknown compiler")

@dataclass
class BenchmarkConfig:
    math_libraries = ["SVML", "SLEEF"]
    llvm_fast_math_flags = ["nnan", "contract", "afn"]

    mod_files: str = ""
    architectures: str = ""
    compilers: str = ""
    external_kernel: bool = False
    nmodl_jit: bool = True
    output_directory: str = "benchmark_output"
    instances: int = 100000000
    experiments: int = 5
    modfile_directory: str = "."
    ext_lib_name: str = "libextkernel.so"
    compiler_flags: dict = field(init=False)
    gpu_target_architecture: str = "sm_70"

    def __post_init__(self):
        with open('compiler_flags.json','r') as fp:
            self.compiler_flags = json.load(fp)


class Benchmark:

    def __init__(self, compiler_config, benchmark_config):
        self.results = {}
        self.compiler_config = compiler_config
        self.benchmark_config = benchmark_config

    def translate_mod_file_to_cpp(self, mod_file):
        """Translate mod file to cpp file that can be
        compiled by the compilers specified and then
        executed by NMODL
        """
        pass

    def compile_llvm_ir_clang(self, llvm_ir_file_path, flags, external_lib_path):
        """Compile LLVM IR file with clang"""
        print("Compiling LLVM IR file with clang")
        compiler_cmd = self.compiler_config.get_compiler_cmd("clang")
        bash_command = [compiler_cmd] + flags.split(" ") + ["./"+llvm_ir_file_path, "-fpic", "-shared", "-o {}".format(external_lib_path)]
        print("Executing command: {} {}".format(compiler_cmd, ' '.join(bash_command)))
        result = subprocess.run(" ".join(bash_command), capture_output=True, text=True, shell=True, env=os.environ.copy())
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        result.check_returncode()

    def translate_mod_file_to_llvm_ir_for_clang(self,
        modfile_str,
        modname,
        compiler,
        architecture,
        math_lib,
        flags):
        """Translate mod file to cpp wrapper file and
        LLVM IR file that can be compiled by the compilers
        specified and then executed by NMODL
        """
        cfg = nmodl.CodeGenConfig()
        cfg.llvm_ir = True
        cfg.llvm_opt_level_ir = 3
        cfg.llvm_math_library = math_lib
        cfg.llvm_fast_math_flags = self.benchmark_config.llvm_fast_math_flags
        cfg.llvm_cpu_name = architecture
        if architecture == "skylake-avx512":
            cfg.llvm_vector_width = 8
        elif architecture == "broadwell":
            cfg.llvm_vector_width = 4
        elif architecture == "nehalem":
            cfg.llvm_vector_width = 2
        else:
            cfg.llvm_vector_width = 1
        cfg.llvm_opt_level_codegen = 3
        if math_lib == "SVML":
            cfg.shared_lib_paths = [self.compiler_config.svml_lib]
        elif math_lib == "SLEEF":
            cfg.shared_lib_paths = [self.compiler_config.sleef_lib]
        cfg.output_dir = str((Path(self.benchmark_config.output_directory)
            / modname
            / compiler
            / architecture
            / self._get_flags_string(flags+"_"+math_lib)))
        modast = self.init_ast(modfile_str)
        # Run JIT to generate the LLVM IR with the wrappers needed to run JIT later
        jit = nmodl.Jit(cfg)
        res = jit.run(modast, modname, 1, 1)
        jit_llvm_ir_file_path = str(Path(cfg.output_dir) / "v{}_{}_opt.ll".format(cfg.llvm_vector_width, modname))
        jit_llvm_ir_file_ext_path = str(Path(cfg.output_dir) / "v{}_{}_opt_ext.ll".format(cfg.llvm_vector_width, modname))
        with open(jit_llvm_ir_file_path, "r") as inf:
            llvm_ir_file_content = inf.read()
            llvm_ir_file_content = re.sub(r'nrn_state_{}'.format(modname.replace('-','_')), r'_Z13nrn_state_extPv', llvm_ir_file_content)
            llvm_ir_file_content = re.sub(r'nrn_cur_{}'.format(modname.replace('-','_')), r'_Z11nrn_cur_extPv', llvm_ir_file_content)
            with open(jit_llvm_ir_file_ext_path, "w") as outf:
                outf.write(llvm_ir_file_content)
        return jit_llvm_ir_file_ext_path

    def _get_flags_string(self, flags):
        return flags.replace(" ", "_").replace('-','').replace('=','_')

    def _make_external_lib_basepath(self, cpp_file, compiler, architecture, flags):
        cpp_basename = os.path.splitext(os.path.basename(cpp_file))[0]
        external_lib_dir = (Path(self.benchmark_config.output_directory)
                / cpp_basename
                / compiler
                / architecture
                / self._get_flags_string(flags))
        if not os.path.exists(external_lib_dir):
            os.makedirs(external_lib_dir)
        return external_lib_dir

    def _get_external_lib_path(self, cpp_file, compiler, architecture, flags):
        external_lib_path = self._make_external_lib_basepath(
                cpp_file, compiler, architecture, flags) / self.benchmark_config.ext_lib_name
        return external_lib_path

    def compile_external_library(self, cpp_file, compiler, architecture, flags):
        """Compile cpp_file to an external shared library
        that has the state and current kernels and can be
        then loaded by NMODL to execute these kernels
        """
        print("Compiling external library with {} compiler ({}, {})".format(compiler, architecture, flags))
        compiler_cmd = self.compiler_config.get_compiler_cmd(compiler)

        cpp_basename = os.path.splitext(os.path.basename(cpp_file))[0]
        external_lib_dir = self._make_external_lib_basepath(cpp_file, compiler, architecture, flags)
        # expsyn mod file openacc execution is diferent than the rest of the mod files
        if compiler == "nvhpc" and architecture == "nvptx64":
            cpp_basename_org = cpp_basename
            cpp_basename = cpp_basename + "_openacc"
            cpp_file_org = cpp_file
            cpp_file = cpp_file.replace(cpp_basename_org, cpp_basename)
            print("Changing {} file to {}".format(cpp_file_org, cpp_file))
        # Replace current cpp_file pragma with correct one and write it in new file
        sed_replaced_cpp_file = external_lib_dir / (Path(cpp_basename + "_ext.cpp"))
        with open(cpp_file, "r") as inf:
            cpp_file_content = inf.read()
            if "-fopenmp" in flags or "-mp=autopar" in flags or compiler == "intel":
                cpp_file_content = re.sub(r'#pragma.*', r'#pragma omp simd', cpp_file_content)
            elif compiler == "clang":
                cpp_file_content = re.sub(r'#pragma.*', r'#pragma clang vectorize(enable)', cpp_file_content)
            elif compiler == "gcc":
                cpp_file_content = re.sub(r'#pragma.*', r'#pragma GCC ivdep', cpp_file_content)
            elif compiler == "nvhpc" and "openacc" not in cpp_basename and architecture == "nvptx64":
                cpp_file_content = re.sub(r'#pragma.*', r'#pragma acc parallel loop deviceptr(inst)', cpp_file_content)
            with open(sed_replaced_cpp_file, "w") as outf:
                outf.write(cpp_file_content)

        if "openacc" in cpp_file:
            cpp_file = cpp_file.replace("_openacc", "")

        external_lib_path = self._get_external_lib_path(cpp_file, compiler, architecture, flags)
        intel_lib_dir = os.path.dirname(self.compiler_config.svml_lib)
        if architecture != "nvptx64":
            bash_command = [compiler_cmd] + flags.split(" ") + ["./"+str(sed_replaced_cpp_file), "-fpic", "-shared", "-o {}".format(external_lib_path), "-Wl,-rpath,{}".format(intel_lib_dir), "-L{}".format(intel_lib_dir), "-lsvml"]
        else:
            bash_command = [compiler_cmd] + flags.split(" ") + ["./"+str(sed_replaced_cpp_file), "-fPIC", "-shared", "-o {}".format(external_lib_path), "-acc", "-nomp", "-gpu=cc70"]
        if "-fopenmp" in flags:
            if compiler == "gcc":
                bash_command.append("-Wl,-rpath,{}".format("/".join(self.compiler_config.gcc_exe.split("/")[0:-2]+["lib64"])))
            elif compiler == "clang":
                bash_command.append("-Wl,-rpath,{}".format("/".join(self.compiler_config.clang_exe.split("/")[0:-2]+["lib"])))
        print("Executing command: {}".format(' '.join(bash_command)))
        result = subprocess.run(" ".join(bash_command), capture_output=True, text=True, shell=True, env=os.environ.copy())
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        result.check_returncode()

    def run_external_kernel(
        self,
        modfile_str,
        modname,
        compiler,
        architecture,
        gpu_target_architecture,
        flags,
        instances,
        experiments,
    ):
        """Runs all external kernel related benchmarks"""
        """Runs NMODL JIT kernels"""
        cfg = nmodl.CodeGenConfig()
        cfg.llvm_ir = True
        cfg.llvm_opt_level_ir = 3
        if architecture != "nvptx64":
            cfg.llvm_math_library = "SVML"
        else:
            cfg.llvm_math_library = "libdevice"
        cfg.llvm_fast_math_flags = self.benchmark_config.llvm_fast_math_flags
        if architecture != "nvptx64":
            cfg.llvm_cpu_name = architecture
        if architecture == "skylake-avx512":
            cfg.llvm_vector_width = 8
        elif architecture == "broadwell":
            cfg.llvm_vector_width = 4
        elif architecture == "nehalem":
            cfg.llvm_vector_width = 2
        else:
            cfg.llvm_vector_width = 1
        cfg.llvm_opt_level_codegen = 3
        if architecture != "nvptx64":
            cfg.shared_lib_paths = [self.compiler_config.svml_lib]
        else:
            cfg.shared_lib_paths = [self.compiler_config.libdevice_lib]
            cfg.llvm_gpu_name = "nvptx64"
            cfg.llvm_gpu_target_architecture = gpu_target_architecture
        cfg.output_dir = str((Path(self.benchmark_config.output_directory)
            / modname
            / compiler
            / architecture
            / self._get_flags_string(flags)))
        modast = self.init_ast(modfile_str)
        jit = nmodl.Jit(cfg)
        external_lib_path = "./" / self._get_external_lib_path(modname+".cpp", compiler, architecture, flags)
        res = jit.run(modast, modname, (int)(experiments), (int)(instances), str(external_lib_path))
        return res

    def run_JIT_kernels(
        self,
        modfile_str,
        modname,
        architecture,
        gpu_target_architecture,
        fast_math_flags,
        math_lib,
        instances,
        experiments,
        gpu_grid_dim = 1,
        gpu_block_dim = 1
    ):
        """Runs NMODL JIT kernels"""
        cfg = nmodl.CodeGenConfig()
        cfg.llvm_ir = True
        cfg.llvm_opt_level_ir = 3
        cfg.llvm_math_library = math_lib
        cfg.llvm_fast_math_flags = fast_math_flags
        if architecture != "nvptx64":
            cfg.llvm_cpu_name = architecture
        else:
            cfg.llvm_gpu_name = "nvptx64"
            cfg.llvm_gpu_target_architecture = gpu_target_architecture
        if architecture == "skylake-avx512":
            cfg.llvm_vector_width = 8
        elif architecture == "broadwell":
            cfg.llvm_vector_width = 4
        elif architecture == "nehalem":
            cfg.llvm_vector_width = 2
        else:
            cfg.llvm_vector_width = 1
        cfg.llvm_opt_level_codegen = 3
        if math_lib == "SVML":
            cfg.shared_lib_paths = [self.compiler_config.svml_lib]
        elif math_lib == "SLEEF":
            cfg.shared_lib_paths = [self.compiler_config.sleef_lib]
        elif math_lib == "libdevice":
            cfg.shared_lib_paths = [self.compiler_config.libdevice_lib]
        cfg.output_dir = str((Path(self.benchmark_config.output_directory)
                / modname
                / "nmodl_jit"
                / architecture
                / math_lib))
        modast = self.init_ast(modfile_str)
        jit = nmodl.Jit(cfg)
        res = jit.run(modast, modname, (int)(experiments), (int)(instances), "", gpu_grid_dim, gpu_block_dim)
        return res

    def init_ast(self, mod_file_string):
        driver = nmodl.NmodlDriver()
        modast = driver.parse_string(mod_file_string)
        return modast

    def run_benchmark(self):
        for modfile in self.benchmark_config.mod_files:
            modname = os.path.splitext(os.path.basename(modfile))[0]
            print("Running benchmark for mod file: {}".format(modfile))
            if modfile not in self.results:
                self.results[modname] = {}
            # Make number of instances smalle for hh kernel due to it's already large memory footprint
            if modname == "hh":
                kernel_instance_size = self.benchmark_config.instances / 5
            else:
                kernel_instance_size = self.benchmark_config.instances

            with open(modfile) as f:
                modfile_str = f.read()

                # Delete existing output directory for mod file
                output_dir = os.path.join(self.benchmark_config.output_directory, modname)
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)

                for architecture in self.benchmark_config.architectures:
                    print('Architecture: {}'.format(architecture))
                    if architecture not in self.results[modname]:
                        self.results[modname][architecture] = {}
                    if self.benchmark_config.external_kernel:
                        for compiler in self.benchmark_config.compilers:
                            # Don't try to use NVPTX64 arch for compilers except NVHPC and other architectures with NVHPC compiler
                            if (architecture == "nvptx64" and compiler != "nvhpc") or (architecture not in ["skylake-avx512", "nvptx64"] and compiler == "nvhpc"):
                                continue
                            if compiler not in self.results[modname][architecture]:
                                self.results[modname][architecture][compiler] = {}
                            for flags in self.benchmark_config.compiler_flags[compiler][
                                architecture
                            ]:
                                # Translate mod file to .cpp to be compiled by the certain compiler
                                # TODO: see above
                                # Compile the .cpp file to a shared library
                                self.compile_external_library(os.path.join("kernels", modname+".cpp"), compiler, architecture, flags)
                                # Run NMODL JIT with external shared library
                                self.results[modname][architecture][compiler][self._get_flags_string(flags)] = self.run_external_kernel(modfile_str,
                                        modname,
                                        compiler,
                                        architecture,
                                        self.benchmark_config.gpu_target_architecture,
                                        flags,
                                        kernel_instance_size,
                                        self.benchmark_config.experiments)
                                print(
                                    "self.results[modname][architecture][compiler][flags] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)"
                                )
                                if compiler == "clang" and "SVML" not in flags:
                                    for math_lib in self.benchmark_config.math_libraries:
                                        # Generate LLVM IR from NMODL JIT
                                        # sed the nrn_state_hh name to _Z16nrn_state_hh_extPv to match the external kernel signature name of the external shared lib
                                        # compile LLVM IR using clang and the compiler flags of the architecture used and generate shared library
                                        # Run NMODL JIT with external shared library
                                        jit_llvm_ir_file_ext_path = self.translate_mod_file_to_llvm_ir_for_clang(modfile_str, modname, compiler, architecture, math_lib, flags+"_jit")
                                        external_lib_path = self._get_external_lib_path(modname, "clang", architecture, flags+"_jit_"+math_lib)
                                        self.compile_llvm_ir_clang(jit_llvm_ir_file_ext_path, flags, external_lib_path)
                                        self.results[modname][architecture][compiler][self._get_flags_string(flags)+"_jit_"+math_lib] = self.run_external_kernel(modfile_str,
                                                modname,
                                                compiler,
                                                architecture,
                                                self.benchmark_config.gpu_target_architecture,
                                                flags+"_jit_"+math_lib,
                                                kernel_instance_size,
                                                self.benchmark_config.experiments)
                                        print(
                                            'self.results[modname][architecture][compiler][flags+"jit"] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)'
                                        )
                    if self.benchmark_config.nmodl_jit:
                        self.results[modname][architecture]["nmodl_jit"] = {}
                        for fast_math in [False, True]:
                            if fast_math:
                                fast_math_flags = self.benchmark_config.llvm_fast_math_flags
                                fast_math_name = "nnancontractafn"
                            else:
                                fast_math_flags = [""]
                                fast_math_name = "nonfastmath"
                            if architecture != "nvptx64":
                                for math_lib in self.benchmark_config.math_libraries:
                                    # Run NMODL JIT on CPU
                                    print(
                                        'self.results[modname][architecture]["nmodl_jit"][math_lib+fast_math_name] = jit.run(modast, modname, self.config.instances, self.config.experiments)'
                                    )
                                    self.results[modname][architecture]["nmodl_jit"][
                                        math_lib + "_" + fast_math_name
                                    ] = self.run_JIT_kernels(
                                        modfile_str,
                                        modname,
                                        architecture,
                                        "",
                                        fast_math_flags,
                                        math_lib,
                                        kernel_instance_size,
                                        self.benchmark_config.experiments
                                    )
                            else:
                                # Run NMODL JIT on GPU
                                self.results[modname][architecture]["nmodl_jit"][
                                    "libdevice_" + fast_math_name
                                ] = self.run_JIT_kernels(
                                    modfile_str,
                                    modname,
                                    architecture,
                                    self.benchmark_config.gpu_target_architecture,
                                    fast_math_flags,
                                    "libdevice",
                                    kernel_instance_size,
                                    self.benchmark_config.experiments,
                                    1024,
                                    128
                                )
                                print(
                                    'self.results[modname][architecture]["nmodl_jit_cuda"]["libdevice"+fast_math_name] = jit.run(modast, modname, self.config.instances, self.config.experiments)'
                                )
                    if not self.benchmark_config.external_kernel and not self.benchmark_config.nmodl_jit:
                        raise Exception("No kernel to run. Select either --external or/and --nmodl_jit")
            print(self.results)
        with open('{}/benchmark_results.pickle'.format(self.benchmark_config.output_directory), 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark script for NMODL LLVM.")
    # Arguments to initialize BenchmarkConfig
    parser.add_argument(
        "--modfiles",
        nargs="+",
        help="Mod files to benchmark",
        required=True,
    )
    parser.add_argument(
        "--architectures", nargs="+", help="Architectures to benchmark", required=True
    )
    parser.add_argument(
        "--compilers", nargs="+", help="Compilers to benchmark", required=True,
        choices=["intel", "clang", "gcc", "nvhpc"]
    )
    parser.add_argument(
        "--external_kernel",
        help="Run external kernel benchmarks",
        action="store_true",
    )
    parser.add_argument(
        "--nmodl_jit",
        help="Run JIT benchmarks with NMODL generated kernels",
        action="store_true",
    )
    parser.add_argument(
        "--output", help="Output directory for benchmark results", required=True
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=100000000,
        help="Instances to benchmark",
        required=False,
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=5,
        help="Experiments to benchmark",
        required=True,
    )
    # Arguments to initialize CompilersConfig
    parser.add_argument(
        "--svml_lib", type=str, help="SVML library directory to use", required=True
    )
    parser.add_argument(
        "--intel_exe", type=str, help="Intel compiler executable to use", required=True
    )
    parser.add_argument(
        "--sleef_lib", type=str, help="Sleef library directory to use", required=True
    )
    parser.add_argument(
        "--clang_exe", type=str, help="Clang compiler executable to use", required=True
    )
    parser.add_argument(
        "--llc_exe", type=str, help="LLC compiler executable to use", required=True
    )
    parser.add_argument(
        "--gcc_exe", type=str, help="GCC compiler executable to use", required=True
    )
    parser.add_argument(
        "--nvhpc_exe", type=str, help="NVHPC compiler executable to use", required=True
    )
    parser.add_argument(
        "--libdevice_lib",
        type=str,
        help="Libdevice library directory to use",
        required=True,
    )
    parser.add_argument(
        "--nmodl_exe", type=str, help="NMODL executable to use", required=True
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    benchmark_config = BenchmarkConfig(
        args.modfiles,
        args.architectures,
        args.compilers,
        args.external_kernel,
        args.nmodl_jit,
        args.output,
        args.instances,
        args.experiments
    )
    compilers_config = CompilersConfig(
        args.svml_lib,
        args.intel_exe,
        args.sleef_lib,
        args.clang_exe,
        args.llc_exe,
        args.gcc_exe,
        args.nvhpc_exe,
        args.libdevice_lib,
        args.nmodl_exe
    )
    benchmark = Benchmark(compilers_config, benchmark_config)
    benchmark.run_benchmark()
    return

if __name__ == "__main__":
    main()

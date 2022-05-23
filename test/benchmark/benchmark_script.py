import argparse
import shutil
import os
import pickle
import subprocess

from matplotlib import pyplot as plt
import seaborn as sns

import nmodl.dsl as nmodl

class CompilersConfig:
    svml_lib = ""
    intel_exe = ""
    sleef_lib = ""
    clang_exe = ""
    llc_exe = ""
    gcc_exe = ""
    libdevice_lib = ""
    nmodl_exe = ""

    def __init__(
        self,
        svml_lib,
        intel_exe,
        sleef_lib,
        clang_exe,
        llc_exe,
        gcc_exe,
        libdevice_lib,
        nmodl_exe,
    ):
        self.svml_lib = svml_lib
        self.intel_exe = intel_exe
        self.sleef_lib = sleef_lib
        self.clang_exe = clang_exe
        self.llc_exe = llc_exe
        self.gcc_exe = gcc_exe
        self.libdevice_lib = libdevice_lib
        self.nmodl_exe = nmodl_exe


class BenchmarkConfig:
    mod_files = ""
    architectures = ""
    compilers = ""
    math_libraries = ["SVML", "SLEEF"]
    fast_math = [False, True]
    llvm_fast_math_flags = ["nnan", "contract", "afn"]
    external_kernel = False
    instances = 100000000
    experiments = 5
    modfile_directory = "."
    output_directory = "benchmark_output"
    ext_lib_name = "libextkernel.so"
    compiler_flags = {}
    compiler_flags["intel"] = {}
    compiler_flags["clang"] = {}
    compiler_flags["gcc"] = {}
    compiler_flags["intel"]["skylake-avx512"] = [
        "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml",
        "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml -fopenmp",
    ]
    compiler_flags["intel"]["broadwell"] = [
        "-O2 -march=broadwell -mtune=broadwell -prec-div -fimf-use-svml"
    ]
    compiler_flags["intel"]["nehalem"] = ["-O2 -msse2 -prec-div -fimf-use-svml"]
    compiler_flags["clang"]["skylake-avx512"] = [
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fveclib=SVML",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp -fveclib=SVML",
    ]
    compiler_flags["clang"]["broadwell"] = [
        "-O3 -march=broadwell -mtune=broadwell -ffast-math -fopenmp",
        "-O3 -march=broadwell -mtune=broadwell -ffast-math -fopenmp -fveclib=SVML",
    ]
    compiler_flags["clang"]["nehalem"] = [
        "-O3 -march=nehalem -mtune=nehalem -ffast-math -fopenmp",
        "-O3 -march=nehalem -mtune=nehalem -ffast-math -fopenmp -fveclib=SVML",
    ]
    compiler_flags["gcc"]["skylake-avx512"] = [
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -fopenmp",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml",
        "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp",
    ]
    compiler_flags["gcc"]["broadwell"] = [
        "-O3 -march=broadwell -mtune=broadwell -ffast-math -ftree-vectorize -fopenmp",
        "-O3 -march=broadwell -mtune=broadwell -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp",
    ]
    compiler_flags["gcc"]["nehalem"] = [
        "-O3 -march=nehalem -mtune=nehalem -ffast-math -ftree-vectorize -fopenmp",
        "-O3 -march=nehalem -mtune=nehalem -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp",
    ]

    def __init__(
        self, mod_files, architectures, compilers, external_kernel, output
    ):
        self.mod_files = mod_files
        self.architectures = architectures
        self.compilers = compilers
        self.external_kernel = external_kernel
        self.output_directory = output

    def __init__(
        self,
        mod_files,
        architectures,
        compilers,
        external_kernel,
        output,
        instances,
        experiments,
    ):
        self.mod_files = mod_files
        self.architectures = architectures
        self.compilers = compilers
        self.external_kernel = external_kernel
        self.output_directory = output
        self.instances = instances
        self.experiments = experiments


class Benchmark:
    benchmark_config = None
    compiler_config = None
    results = {}

    def __init__(self, compiler_config, benchmark_config):
        self.compiler_config = compiler_config
        self.benchmark_config = benchmark_config

    def translate_mod_file_to_cpp(self, mod_file):
        """Translate mod file to cpp file that can be
        compiled by the compilers specified and then
        executed by NMODL
        """
        pass

    def compile_external_library(self, cpp_file, compiler, architecture, flags):
        """Compile cpp_file to an external shared library
        that has the state and current kernels and can be
        then loaded by NMODL to execute these kernels
        """
        print("Compiling external library with {} compiler ({}, {})".format(compiler, architecture, flags))
        compiler_cmd = ""
        if compiler == "intel":
            compiler_cmd = self.compiler_config.intel_exe
        elif compiler == "clang":
            compiler_cmd = self.compiler_config.clang_exe
        elif compiler == "gcc":
            compiler_cmd = self.compiler_config.gcc_exe
        else:
            raise Exception("Unknown compiler")
        
        external_lib_dir = os.path.join(self.benchmark_config.output_directory, cpp_file.split("/")[-1].split(".")[0], compiler, architecture, flags.replace(" ", "_").replace('-','').replace('=',''))
        if not os.path.exists(external_lib_dir):
            os.makedirs(external_lib_dir)
        external_lib_path = os.path.join(external_lib_dir, self.benchmark_config.ext_lib_name)
        intel_lib_dir = '/'.join(self.compiler_config.svml_lib.split("/")[0:-1])
        bash_command = [compiler_cmd] + flags.split(" ") + ["./"+cpp_file, "-shared", "-fpic", "-o {}".format(external_lib_path), "-Wl,-rpath,{}".format(intel_lib_dir), "-L{}".format(intel_lib_dir), "-lsvml"]
        print("Executing command: {}".format(bash_command))
        process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if error:
            raise Exception(error)
        print("Compilation output of shared ext kernel lib with {} compiler: {}".format(compiler, output))

    def run_external_kernel(
        self,
        modfile_str,
        modname,
        compiler,
        architecture,
        flags,
        instances,
        experiments,
    ):
        """Runs all external kernel related benchmarks"""
        """Runs NMODL JIT kernels"""
        cfg = nmodl.CodeGenConfig()
        cfg.llvm_ir = True
        cfg.llvm_opt_level_ir = 3
        cfg.llvm_math_library = "SVML"
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
        cfg.shared_lib_paths = [self.compiler_config.svml_lib]
        cfg.output_dir = os.path.join(self.benchmark_config.output_directory, modname, architecture, math_lib)
        modast = self.init_ast(modfile_str)
        jit = nmodl.Jit(cfg)
        external_lib_path = os.path.join(self.benchmark_config.output_directory, modname, compiler, architecture, flags.replace(" ", "_"), self.benchmark_config.ext_lib_name)
        res = jit.run(modast, modname, (int)(experiments), (int)(instances), external_lib_path)
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
        cfg.output_dir = os.path.join(self.benchmark_config.output_directory, modname, architecture, math_lib)
        modast = self.init_ast(modfile_str)
        jit = nmodl.Jit(cfg)
        res = jit.run(modast, modname, (int)(experiments), (int)(instances))
        return res

    def init_ast(self, mod_file_string):
        driver = nmodl.NmodlDriver()
        modast = driver.parse_string(mod_file_string)
        return modast

    def run_benchmark(self):
        for modfile in self.benchmark_config.mod_files:
            modname = modfile.split("/")[-1].split(".")[0]
            print("Running benchmark for mod file: {}".format(modfile))
            if modfile not in self.results:
                self.results[modname] = {}
            # Make number of instances smalle for hh kernel due to it's already large memory footprint
            if [modfile == "hh.mod"]:
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
                                print(
                                    "self.results[modname][architecture][compiler][flags] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)"
                                )
                                if compiler == "clang":
                                    for (
                                        math_lib
                                    ) in self.benchmark_config.math_libraries:
                                        # Generate LLVM IR from NMODL JIT
                                        # sed the nrn_state_hh name to _Z16nrn_state_hh_extPv to match the external kernel signature name of the external shared lib
                                        # compile LLVM IR using clang and the compiler flags of the architecture used and generate shared library
                                        # Run NMODL JIT with external shared library
                                        print(
                                            'self.results[modname][architecture][compiler][flags+"jit"] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)'
                                        )
                    self.results[modname][architecture]["nmodl_jit"] = {}
                    for fast_math in self.benchmark_config.fast_math:
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
                                    self.benchmark_config.experiments,
                                )
                        else:
                            # Run NMODL JIT on GPU
                            print(
                                'self.results[modname][architecture]["nmodl_jit_cuda"]["libdevice"+fast_math_name] = jit.run(modast, modname, self.config.instances, self.config.experiments)'
                            )
            print(self.results)
            with open('benchmark_results.pickle', 'wb') as handle:
                pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_results(self, file = ""):
        if file != "":
            with open(file, 'rb') as handle:
                self.results = pickle.load(handle)
        # plot results in bar for each mod file, architecture and flags with matplotlib
        for modname in self.results:
            bar_labels = []
            bar_data_state = {}
            bar_data_cur = {}
            for architecture in self.results[modname]:
                for compiler in self.results[modname][architecture]:
                    if compiler == "nmodl_jit" or compiler == "nmodl_jit_cuda":
                        for math_lib_fast_math_flag in self.results[modname][architecture][compiler]:
                            dict_label = "{}_{}_{}".format(
                                    architecture, compiler, math_lib_fast_math_flag
                                )
                            bar_data_state[dict_label] = self.results[modname][architecture][compiler][math_lib_fast_math_flag]["nrn_state_hh"][0]
                            bar_data_cur[dict_label] = self.results[modname][architecture][compiler][math_lib_fast_math_flag]["nrn_cur_hh"][0]
            keys = list(bar_data_state.keys())
            state_vals = [float(bar_data_state[k]) for k in keys]
            cur_vals = [float(bar_data_cur[k]) for k in keys]
            state_barplot = sns.barplot(x=keys, y=state_vals)
            for item in state_barplot.get_xticklabels():
                item.set_rotation(90)
            plt.savefig("{}_state_benchmark.pdf".format(modname), format="pdf", bbox_inches="tight")
            plt.close()
            cur_barplot = sns.barplot(x=keys, y=cur_vals)
            for item in cur_barplot.get_xticklabels():
                item.set_rotation(90)
            plt.savefig("{}_cur_benchmark.pdf".format(modname), format="pdf", bbox_inches="tight")
            plt.close()


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
        "--compilers", nargs="+", help="Compilers to benchmark", required=True
    )
    parser.add_argument(
        "--external_kernel",
        help="Run external kernel benchmarks",
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
        args.output,
        args.instances,
        args.experiments,
    )
    compilers_config = CompilersConfig(
        args.svml_lib,
        args.intel_exe,
        args.sleef_lib,
        args.clang_exe,
        args.llc_exe,
        args.gcc_exe,
        args.libdevice_lib,
        args.nmodl_exe,
    )
    benchmark = Benchmark(compilers_config, benchmark_config)
    benchmark.run_benchmark()
    benchmark.plot_results()
    return

if __name__ == "__main__":
    main()

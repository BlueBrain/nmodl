import argparse

class CompilersConfig():
    svml_lib = ""
    intel_exe = ""
    sleef_lib = ""
    clang_exe = ""
    llc_exe = ""
    gcc_exe = ""
    libdevice_lib = ""
    nmodl_exe = ""
    def __init__(self, svml_lib, intel_exe, sleef_lib, clang_exe, llc_exe, gcc_exe, libdevice_lib, nmodl_exe):
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
    llvm_fast_math_flags = "--fmf nnan contract afn"
    external_kernel = False
    instances = 100000000
    experiments = 5
    modfile_directory = "."
    output_directory = "benchmark_output"
    ext_lib_name = "libextkernel.so"
    compiler_flags = {}
    compiler_flags["intel"]["skylake_avx512"] = [
        "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml",
        "-O2 -march=skylake-avx512 -mtune=skylake -prec-div -fimf-use-svml -fopenmp",
    ]
    compiler_flags["intel"]["broadwell"] = [
        "-O2 -march=broadwell -mtune=broadwell -prec-div -fimf-use-svml"
    ]
    compiler_flags["intel"]["nehalem"] = ["-O2 -msse2 -prec-div -fimf-use-svml"]
    compiler_flags["clang"]["skylake_avx512"] = [
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
    compiler_flags["gcc"]["skylake_avx512"] = [
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
        self, mod_files, architectures, compilers, external_kernel, output, modfile_dir
    ):
        self.mod_files = mod_files
        self.architectures = architectures
        self.compilers = compilers
        self.external_kernel = external_kernel
        self.output_directory = output
        self.modfile_directory = modfile_dir

    def __init__(
        self,
        mod_files,
        architectures,
        compilers,
        external_kernel,
        output,
        modfile_dir,
        instances,
        experiments,
    ):
        self.mod_files = mod_files
        self.architectures = architectures
        self.compilers = compilers
        self.external_kernel = external_kernel
        self.output_directory = output
        self.modfile_directory = modfile_dir
        self.instances = instances
        self.experiments = experiments


class Benchmark:
    benchmark_config = None
    compiler_config = None
    results = {}

    def __init__(self, compiler_config, benchmark_config):
        self.compiler_config = compiler_config
        self.benchmark_config = benchmark_config

    def run_external_kernel(self):
        """Runs all external kernel related benchmarks"""
        pass

    def run_JIT_kernels(self):
        """Runs NMODL JIT kernels"""
        pass

    def run_benchmark(self):
        for modfile in self.config.mod_files:
            if modfile not in self.results:
                self.results[modfile] = {}
            for architecture in self.config.architectures:
                if architecture not in self.results[modfile]:
                    self.results[modfile][architecture] = {}
                if self.config.external_kernel:
                    for compiler in self.config.compilers:
                        if compiler not in self.results[modfile][architecture]:
                            self.results[modfile][architecture][compiler] = {}
                        for flags in self.config.compiler_flags[compiler][architecture]:
                            # Translate mod file to .cpp to be compiled by the certain compiler
                            # Compile the .cpp file to a shared library
                            # Run NMODL JIT with external shared library
                            print(
                                "self.results[modfile][architecture][compiler][flags] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)"
                            )
                            if compiler == "clang":
                                for math_lib in self.config.math_libraries:
                                    # Generate LLVM IR from NMODL JIT
                                    # sed the nrn_state_hh name to _Z16nrn_state_hh_extPv to match the external kernel signature name of the external shared lib
                                    # compile LLVM IR using clang and the compiler flags of the architecture used and generate shared library
                                    # Run NMODL JIT with external shared library
                                    print(
                                        'self.results[modfile][architecture][compiler][flags+"jit"] = jit.run(modast, modname, self.config.instances, self.config.experiments, external_lib)'
                                    )
                for fast_math in self.config.fast_math:
                    if fast_math:
                        fast_math_flags = self.llvm_fast_math_flags
                        fast_math_name = "nnancontractafn"
                    else:
                        fast_math_flags = ""
                        fast_math_name = "nonfastmath"
                    if architecture != "nvptx64":
                        for math_lib in self.config.math_libraries:
                            # Run NMODL JIT on CPU
                            print(
                                'self.results[modfile][architecture]["nmodl_jit"][math_lib+fast_math_name] = jit.run(modast, modname, self.config.instances, self.config.experiments)'
                            )
                    else:
                        # Run NMODL JIT on CPU
                        print(
                            'self.results[modfile][architecture]["nmodl_jit_cuda"]["libdevice"+fast_math_name] = jit.run(modast, modname, self.config.instances, self.config.experiments)'
                        )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark script for NMODL LLVM.')
    # Arguments to initialize BenchmarkConfig
    parser.add_argument(
        "--modfiles",
        nargs="+",
        help="Mod files to benchmark",
        required=True,
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        help="Architectures to benchmark",
        required=True
    )
    parser.add_argument(
        "--compilers",
        nargs="+",
        help="Compilers to benchmark",
        required=True
    )
    parser.add_argument(
        "--external_kernel",
        help="Run external kernel benchmarks",
        action="store_true",
        required=True
    )
    parser.add_argument(
        "--output",
        help="Output directory for benchmark results",
        required=True
    )
    parser.add_argument(
        "--modfile_dir",
        help="Directory containing mod files",
        required=True
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=100000000,
        help="Instances to benchmark",
        required=False
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
        "--svml_lib",
        type=str,
        help="SVML library directory to use",
        required=True
    )
    parser.add_argument(
        "--intel_exe",
        type=str,
        help="Intel compiler executable to use",
        required=True
    )
    parser.add_argument(
        "--sleef_lib",
        type=str,
        help="Sleef library directory to use",
        required=True
    )
    parser.add_argument(
        "--clang_exe",
        type=str,
        help="Clang compiler executable to use",
        required=True
    )
    parser.add_argument(
        "--llc_exe",
        type=str,
        help="LLC compiler executable to use",
        required=True
    )
    parser.add_argument(
        "--gcc_exe",
        type=str,
        help="GCC compiler executable to use",
        required=True
    )
    parser.add_argument(
        "--libdevice_lib",
        type=str,
        help="Libdevice library directory to use",
        required=True
    )
    parser.add_argument(
        "--nmodl_exe",
        type=str,
        help="NMODL executable to use",
        required=True
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
        args.modfile_dir,
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
        args.libdevice_lib,
        args.nmodl_exe
    )
    benchmark = Benchmark(benchmark_config, compilers_config)
    benchmark.run_benchmark()
    return

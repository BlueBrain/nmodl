
# MOD2IR: High-Performance Code Generation for a Biophysically Detailed Neuronal Simulation DSL

## Artifact Description

MOD2IR is implemented as a code generation backend inside the NMODL Framework and it makes heavy
use of the LLVM IR and compilation passes. Most of the relevant code of the described work can be
found [here](https://github.com/BlueBrain/nmodl/tree/llvm/src/codegen/llvm) and
[here](https://github.com/BlueBrain/nmodl/tree/llvm/test/benchmark).

### Hardware Requirements

The provided artifact can in theory be run on any x86 hardware platform. For the purpose of closely
reproducing our benchmark results it is required a workstation (or cloud instance) with Intel Xeon
Skylake (or newer) CPU that supports AVX-512 instructions and an NVIDIA Volta V100 (or newer) GPU.
All benchmark runs are single-core and have relatively low memory-requirement. For building or running
the Docker image (and more specifically the NMODL Framework) we, however, recommend a system with plenty
of cores, at least 32GB of RAM available and 20 GB of disk space.

### Software Requirements

Any reasonably up-to-date Linux system with Docker should be sufficient (we recommend Ubuntu 22.04). If GPU results are to be
reproduced, an up-to-date CUDA (11.0 or newer) should be present.

## Benchmarking Instructions

To reproduce as closely as possible our environment and to lower the burden of the
installation of the different compilers and libraries we have created Docker images which take
care of installing all the necessary packages and compilers to install MOD2IR and execute the
benchmarks.
Due to technical restrictions imposed by Docker to execute a Docker image and be able to execute
applications on NVIDIA GPUs there are some extra steps needed. For this reason we have created two
different `Dockerfile`s, one that takes care of both the CPU and GPU benchmarks and one for CPU only
execution if there is no NVIDIA GPU available in the test system.

The original instructions (markdown file) with the executable script snippets can be found [here](https://github.com/BlueBrain/nmodl/blob/a65b15ca3edf9f069adb83bb78b9611d58b15a58/docs/CC2023/README.md).

### CPU and GPU docker image

The image that targets both CPU and GPU can be found in `test/benchmark/gpu_docker/Dockerfile`.
To launch the Docker image you can execute the following:

```
git clone -b mod2ir-CC2023 --depth 1 \
   https://github.com/BlueBrain/nmodl.git
# Enter the directory that contains the NVIDIA docker
# runtime installation script
cd nmodl/test/benchmark
# Installs docker and NVIDIA docker runtime (needs sudo
# permission and is based on Ubuntu 22.04 but with small
# changes it should be supported by any Ubuntu version
# or other linux distributions)
bash install_gpu_docker_env.sh
# Execute docker image (~16GB)
docker run -it -v $PWD:/opt/mount --gpus \
   all bluebrain/nmodl:mod2ir-gpu-benchmark
```

After building and launching the docker file we can now execute the benchmarks and generate the same
plots as the ones we included in the paper with the new results along the reference plots from the paper.
To do this we need to execute the following two scripts inside the docker image environment:

```
# Enter the directory where the scripts are inside the docker
# image
cd nmodl/test/benchmark
# Runs all the benchmarks on CPU and GPU
bash run_benchmark_script_cpu_gpu.sh
# Generate the plots based on the outputs of the previous script
python3 plot_benchmarks_cpu_gpu.py
# Copy the graphs from the docker image to your environment
cp -r graphs_output_pandas /opt/mount
```

Executing `run_benchmark_script_dockerfile.sh` will generate two pickle files that include the results
in `hh_expsyn_cpu/benchmark_results.pickle` for the CPU benchmarks and `hh_expsyn_gpu/benchmark_results.pickle`
for the GPU benchmarks. Those will then be loaded by `plot_benchmarks.py` to generate the plots.
Now you can exit the docker image terminal and open the above files which exist in your local directory.


### CPU only docker image

In case there is no GPU available instead of running the above Docker container you can also run a
CPU only container.
To do this you need to:

```
# Execute docker image (~16GB)
docker run -it -v $PWD:/opt/mount \
   bluebrain/nmodl:mod2ir-cpu-benchmark
```

Then inside the docker shell:

```
# Enter the directory where the scripts are inside the docker
# image
cd nmodl/test/benchmark
# Runs all the benchmarks on CPU
bash run_benchmark_script_cpu_only.sh
# Generate the plots based on the outputs of the previous script
python3 plot_benchmarks_cpu_only.py
# Copy the graphs from the docker image to your environment
cp -r graphs_output_pandas /opt/mount
```

By executing `run_benchmark_script_cpu_only.sh` there will be only `hh_expsyn_cpu/benchmark_results.pickle`
generated containing the CPU results.


## Notes

1. The benchmark results with `GCC` and `NVHPC` compilers might be better in the docker container than the presented manuscript due to the newer OS we’re using in the Dockerfile. The current Ubuntu versions come with GLIBC 2.3x, which includes `libmvec` providing vectorized implementations for some mathematical functions and which is used by the `GCC` and `NVHPC` compilers. This results in much more vectorization use in the benchmark kernels even without providing the SVML library to `GCC` (or `NVHPC`).
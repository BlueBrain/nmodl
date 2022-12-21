
# MOD2IR: High-Performance Code Generation for a Biophysically Detailed Neuronal Simulation DSL

## Preliminary Artifact Description

### Broad Description

This artifact provides all the necessary code, scripts and results to compile the NMODL transpiler
with the MOD2IR extension and run all benchmarks described in the manuscript. To simplify the
evaluation process we provide along with the instructions a Dockerfile that will setup a viable
system for the benchmarks. The driver script compiles the membrane mechanism model `hh.mod` and the
synapse mechanism model `expsyn.mod` with various compile-time configurations and then runs the
generated  binaries comparing their runtimes. More specifically the benchmark compares the execution
runtime of the binaries generated via the two-step compilation process MOD-C++-binary using various
open-source and commercial compiler frameworks with the one-step ahead-of-time and just-in-time
processes of MOD2IR.
MOD2IR is implemented as a code generation backend inside the NMODL Framework and it makes heavy
use of the LLVM IR and compilation passes. Most of the relevant code of the described work can be
found [here](https://github.com/BlueBrain/nmodl/tree/llvm/src/codegen/llvm) and
[here](https://github.com/BlueBrain/nmodl/tree/llvm/test/benchmark).

### Badge

Blue Badge (results validated). We hope that using the provided Dockerfile and scripts the
evaluators should be able to fully build our code and reproduce our benchmark setup as well as
obtain benchmarking results. Please note that in all likelihood the obtained runtimes by the
evaluators will slightly differ from the presented results from the paper as they heavily depend on
the used hardware and system software. We believe, however, that the results should nevertheless be
qualitiatively the same as the ones we have presented.

### Hardware requisites

The provided artifact can in theory be run on any x86 hardware platform. For the prupose of closely
reproducing our benchmark results we recommend using a workstation (or cloud instance) with Intel Xeon
Skylake (or newer) CPU and an NVIDIA Volta V100 or (newer GPU). All benchmark runs are single-core
and have relatively low memory-requirement. For building the Docker image (and more specifically the
NMODL Framework) we, however, recommend a system with plenty of cores and at least 32GB of RAM
available and 40 GB of disk space.

### Software requisites

Any reasonably up-to-date Linux system with Docker should be sufficient. If GPU results are to be
reproduced, an up-to-date CUDA (11.0 or newer) should be present.

### Benchmarking Instructions

To reproduce as closely as possible to reproduce our environment and to lower the burden of the
installation of the different compilers and libraries we have created a `Dockerfile` which takes
care of installing all the necessary packages and compilers to install MOD2IR and execute the
benchmarks.
Due to technical restrictions imposed by Docker to execute a Docker image and be able to execute
applications on NVIDIA GPUs there are some extra steps needed. For this reason we have created two
different `Dockerfile`s, one that takes care of both the CPU and GPU benchmarks and one for CPU only
execution if there is no NVIDIA GPU available in the test system.

#### CPU and GPU docker image
The first one that targets both CPU and GPU can be found in `test/benchmark/gpu_docker/Dockerfile`.
To build the image and be able to execute the benchmarks on CPU and GPU it's needed to:
```
cd test/benchnark/gpu_docker  # Enter the directory that contains the Dockerfile
bash install_gpu_docker_env.sh  # Installs docker and NVIDIA docker runtime (needs sudo permission)
docker build nmodl-benchmark:gpu .  # Build docker image (~36GB)
docker run -it -v $PWD:/opt/mount --gpus all nmodl-benchmark:gpu  # Execute docker image
```
After building launching the docker file we can now execute the benchmarks and generate the same
plots as the ones we included in the paper with the new results along the reference plots we actually
have in the paper.
To do the above we need to execute the following two scripts inside the docker image environment:
```
cd nmodl/test/benchmark  # Enter the directory where the scripts are inside the docker image
bash run_benchmark_script_cpu_gpu.sh  # Runs all the benchmarks on CPU and GPU
python plot_benchmarks_cpu_gpu.py  # Generate the plots based on the outputs of the previous script
cp -r graphs_output_pandas /opt/mount  # Copy the graphs from the docker image to your environment
```
Executing `run_benchmark_script_dockerfile.sh` will generate two pickle files that include the results
in `hh_expsyn_cpu/benchmark_results.pickle` for the CPU benchmarks and `hh_expsyn_gpu/benchmark_results.pickle`
for the GPU benchmarks. Those will then be loaded by `plot_benchmarks.py` to generate the plots.
Now you can exit the docker image terminal and open the above files which exist in your local directory.


#### CPU only docker image
In case there is no GPU available instead of building the above docker image you can also use the
`Dockerfile` found in `test/benchmark/cpu_docker/Dockerfile`.
To build this you need to:
```
cd test/benchnark/cpu_docker  # Enter the directory that contains the Dockerfile
docker build nmodl-benchmark:cpu .  # Build docker image (~36GB)
docker run -it -v $PWD:/opt/mount nmodl-benchmark:cpu  # Execute docker image
```
Then inside the docker terminal:
```
cd nmodl/test/benchmark  # Enter the directory where the scripts are inside the docker image
bash run_benchmark_script_cpu_only.sh  # Runs all the benchmarks on CPU
python plot_benchmarks_cpu_only.py  # Generate the plots based on the outputs of the previous script
cp -r graphs_output_pandas /opt/mount  # Copy the graphs from the docker image to your environment
```
By executing `run_benchmark_script_cpu_only.sh` there will be only `hh_expsyn_cpu/benchmark_results.pickle`
generated containing the CPU results.


### Expectations

The expected time for building the docker image is around 10 minutes using a modern multicore system
with a stable internet connection.
The expected runtime of the benchmarks is around 3 hours.

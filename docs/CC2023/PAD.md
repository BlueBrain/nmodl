
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
evaluators will slightly differ from the presented results in the paper as they heavily depend on
the used hardware and system software. We believe, however, that the results should nevertheless be
qualitiatively the same as the ones we have presented.

### Hardware requisites

The provided artifact can in theory be run on any x86 hardware platform. For the prupose of closely
reproducing our benchmark results we recommend using a workstation (or cloud instance) with Intel Xeon
Skylake (or newer) CPU and an NVIDIA Volta V100 (or newer) GPU. All benchmark runs are single-core
and have relatively low memory-requirement. For building the Docker image (and more specifically the
NMODL Framework) we, however, recommend a system with plenty of cores and at least 32GB of RAM
available and 20 GB of disk space.

### Software requisites

Any reasonably up-to-date Linux system with Docker should be sufficient. If GPU results are to be
reproduced, an up-to-date CUDA (11.0 or newer) should be present.


### Expectations

We expect that all setup and benchmarks can be completed within one working day. The expected time for building
the docker image is around 10 minutes using a modern multicore system with a stable internet connection.
The expected runtime of the benchmarks is around 4 hours.

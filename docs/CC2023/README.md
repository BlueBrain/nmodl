
# MOD2IR: High-Performance Code Generation for a Biophysically Detailed Neuronal Simulation DSL

Please refer to the PAD.md file for an overview and necessary prerequisites.

## Benchmarking Instructions

To reproduce as closely as possible our environment and to lower the burden of the
installation of the different compilers and libraries we have created Docker images which take
care of installing all the necessary packages and compilers to install MOD2IR and execute the
benchmarks.
Due to technical restrictions imposed by Docker to execute a Docker image and be able to execute
applications on NVIDIA GPUs there are some extra steps needed. For this reason we have created two
different `Dockerfile`s, one that takes care of both the CPU and GPU benchmarks and one for CPU only
execution if there is no NVIDIA GPU available in the test system.

### CPU and GPU docker image

The image that targets both CPU and GPU can be found in `test/benchmark/gpu_docker/Dockerfile`.
To launch the Docker image you can execute the following:

```
cd test/benchnark/gpu_docker  # Enter the directory that contains the Dockerfile
bash install_gpu_docker_env.sh  # Installs docker and NVIDIA docker runtime (needs sudo permission)
docker run -it -v $PWD:/opt/mount --gpus all bluebrain/nmodl:mod2ir-gpu-benchmark # Execute docker image (~16GB)
```

After building and launching the docker file we can now execute the benchmarks and generate the same
plots as the ones we included in the paper with the new results along the reference plots from the paper.
To do this we need to execute the following two scripts inside the docker image environment:

```
cd nmodl/test/benchmark  # Enter the directory where the scripts are inside the docker image
bash run_benchmark_script_cpu_gpu.sh  # Runs all the benchmarks on CPU and GPU
python3 plot_benchmarks_cpu_gpu.py  # Generate the plots based on the outputs of the previous script
cp -r graphs_output_pandas /opt/mount  # Copy the graphs from the docker image to your environment
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
cd test/benchnark/cpu_docker  # Enter the directory that contains the Dockerfile
docker run -it -v $PWD:/opt/mount bluebrain/nmodl:mod2ir-cpu-benchmark # Execute docker image (~16GB)
```

Then inside the docker shell:

```
cd nmodl/test/benchmark  # Enter the directory where the scripts are inside the docker image
bash run_benchmark_script_cpu_only.sh  # Runs all the benchmarks on CPU
python3 plot_benchmarks_cpu_only.py  # Generate the plots based on the outputs of the previous script
cp -r graphs_output_pandas /opt/mount  # Copy the graphs from the docker image to your environment
```

By executing `run_benchmark_script_cpu_only.sh` there will be only `hh_expsyn_cpu/benchmark_results.pickle`
generated containing the CPU results.


## Notes

1. Acceleration results with `GCC` compiler might be better in the docker container than the paper
   due to the newer OS we're using in the Dockerfile. Latest Ubuntu versions come with GLIBC 2.3x that
   includes `libmvec` which provides vectorized implementations to the `GCC` compiler enabling the
   vectorization of the kernels even without providing the `SVML` library to `GCC`.

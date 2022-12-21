#!/usr/bin/python3

import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns


def _get_flags_string(flags):
    return flags.replace(" ", "_").replace('-','').replace('=','_')

def load_pickle_result_file(pickle_files, results):
    def _merge(a, b, path=None):
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    _merge(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass  # same leaf value
                else:
                    raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a

    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as handle:
            results = _merge(results, pickle.load(handle))
    return results


def generate_graph_pandas_combined_relative_log(
    results,
    compilers_comparison_config,
    graph_suffix,
    output_dir,
    print_values=False,
    xaxis_label=None,
    plot_size=(12, 6),
    baseline_name="intel_svml",
    reference=False,
):
    os.makedirs(output_dir, exist_ok=True)
    compiler_flags = json.loads(compilers_comparison_config)
    ref_title_str = reference ? " (reference)" : ""
    fig, axes = plt.subplots(1, 3, squeeze=False, figsize=plot_size)
    ax_index = 0
    for modname in results:
        # state
        bar_data_state_cpu_panda = {}
        bar_data_state_cpu_panda["architecture"] = []
        bar_data_state_cpu_panda["compiler"] = []
        bar_data_state_cpu_panda["runtime"] = []
        # current
        bar_data_cur_cpu_panda = {}
        bar_data_cur_cpu_panda["architecture"] = []
        bar_data_cur_cpu_panda["compiler"] = []
        bar_data_cur_cpu_panda["runtime"] = []
        baseline_cur = 0.0
        for architecture in results[modname]:
            for compiler in compiler_flags:
                if (
                    compiler in results[modname][architecture]
                    and architecture in compiler_flags[compiler]
                ):
                    for flags in compiler_flags[compiler][architecture]:
                        if compiler == "nmodl_jit":
                            state_kernel_name = "nrn_state_{}".format(
                                modname.replace("-", "_")
                            )
                            cur_kernel_name = "nrn_cur_{}".format(
                                modname.replace("-", "_")
                            )
                        else:
                            state_kernel_name = "nrn_state_ext"
                            cur_kernel_name = "nrn_cur_ext"
                        if compiler == "clang" and "jit" in flags:
                            compiler_name = "mod2ir"
                        elif compiler == "nmodl_jit":
                            compiler_name = "mod2ir_jit"
                        else:
                            compiler_name = compiler
                        if "svml" in flags or "SVML" in flags:
                            compiler_name = compiler_name + "_svml"
                            if architecture != "nvptx64" and compiler == "intel":
                                baseline_state = results[modname][architecture][
                                    "intel"
                                ][_get_flags_string(flags)][state_kernel_name][0]
                                baseline_cur = results[modname][architecture]["intel"][
                                    _get_flags_string(flags)
                                ][cur_kernel_name][0]
                        elif "sleef" in flags or "SLEEF" in flags:
                            compiler_name = compiler_name + "_sleef"
                        if architecture == "default":
                            architecture_label = "auto-scalar"
                        elif architecture == "nehalem":
                            architecture_label = "nehalem-sse2"
                        elif architecture == "broadwell":
                            architecture_label = "broadwell-avx2"
                        elif architecture == "nvptx64":
                            architecture_label = architecture
                            if compiler == "nvhpc":
                                baseline_state = results[modname][architecture][
                                    "nvhpc"
                                ][_get_flags_string(flags)][state_kernel_name][0]
                                baseline_cur = results[modname][architecture]["nvhpc"][
                                    _get_flags_string(flags)
                                ][cur_kernel_name][0]
                        else:  # skylake-avx512
                            architecture_label = architecture
                        if modname != "expsyn":
                            bar_data_state_cpu_panda["architecture"].append(
                                architecture_label
                            )
                            bar_data_state_cpu_panda["compiler"].append(compiler_name)
                            if (
                                _get_flags_string(flags)
                                not in results[modname][architecture][compiler]
                            ):
                                bar_data_state_cpu_panda["runtime"].append(0)
                            else:
                                bar_data_state_cpu_panda["runtime"].append(
                                    results[modname][architecture][compiler][
                                        _get_flags_string(flags)
                                    ][state_kernel_name][0]
                                )
                        bar_data_cur_cpu_panda["architecture"].append(
                            architecture_label
                        )
                        bar_data_cur_cpu_panda["compiler"].append(compiler_name)
                        if (
                            _get_flags_string(flags)
                            not in results[modname][architecture][compiler]
                        ):
                            bar_data_cur_cpu_panda["runtime"].append(0)
                        else:
                            bar_data_cur_cpu_panda["runtime"].append(
                                results[modname][architecture][compiler][
                                    _get_flags_string(flags)
                                ][cur_kernel_name][0]
                            )
        for i, runtime in enumerate(bar_data_state_cpu_panda["runtime"]):
            bar_data_state_cpu_panda["runtime"][i] = baseline_state / runtime
        for i, runtime in enumerate(bar_data_cur_cpu_panda["runtime"]):
            bar_data_cur_cpu_panda["runtime"][i] = baseline_cur / runtime
        pd.options.display.float_format = "{:,.2f}".format
        if modname != "expsyn":
            df_state = pd.DataFrame(
                bar_data_state_cpu_panda,
                columns=["architecture", "compiler", "runtime"],
            )
            print(df_state, type(df_state))
            sns.barplot(
                x="architecture",
                y="runtime",
                hue="compiler",
                data=df_state,
                ax=axes[0, ax_index],
            )
            axes[0, ax_index].set_yscale("symlog", base=2, linthresh=0.015)
            axes[0, ax_index].set_ylim(0.125, 2)
            axes[0, ax_index].set_yticks(
                [0.125, 0.25, 0.5, 1, 2], [0.125, 0.25, 0.5, 1, 2]
            )
            axes[0, ax_index].axhline(1.0, ls="--", color="black")
            axes[0, ax_index].xaxis.label.set_visible(False)
            axes[0, ax_index].yaxis.label.set_visible(False)
            axes[0, ax_index].set_title(f"nrn_state_{modname}{ref_title_str}")
            axes[0, ax_index].get_legend().remove()
            if xaxis_label is not None:
                axes[0, ax_index].get_xaxis().set_visible(False)
            if print_values:
                for i in axes[0, ax_index].containers:
                    axes[0, ax_index].bar_label(
                        i,
                    )
            ax_index += 1
        df_cur = pd.DataFrame(
            bar_data_cur_cpu_panda, columns=["architecture", "compiler", "runtime"]
        )
        ax = sns.barplot(
            x="architecture",
            y="runtime",
            hue="compiler",
            data=df_cur,
            ax=axes[0, ax_index],
        )
        axes[0, ax_index].axhline(1.0, ls="--", color="black")
        print(df_cur, type(df_cur))
        axes[0, ax_index].set_yscale("symlog", base=2, linthresh=0.015)
        axes[0, ax_index].set_ylim(0.125, 2)
        axes[0, ax_index].set_yticks([0.125, 0.25, 0.5, 1, 2], [0.125, 0.25, 0.5, 1, 2])
        axes[0, ax_index].xaxis.label.set_visible(False)
        axes[0, ax_index].yaxis.label.set_visible(False)
        axes[0, ax_index].set_title(f"nrn_cur_{modname}{ref_title_str}")
        axes[0, ax_index].get_legend().remove()
        if xaxis_label is not None:
            axes[0, ax_index].get_xaxis().set_visible(False)
        if print_values:
            for i in axes[0, ax_index].containers:
                axes[0, ax_index].bar_label(
                    i,
                )
        ax_index += 1

    fig.text(
        0.06,
        0.5,
        "Speedup relative to {}".format(baseline_name),
        ha="center",
        va="center",
        rotation="vertical",
    )
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(
        "{}/combined_benchmark_{}.pdf".format(output_dir, graph_suffix),
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_cpu_results():
    colors = [
        "#6baed6",  # intel
        "#0570b0",  # intel svml
        "#66c2a4",  # gcc
        "#238b45",  # gcc svml
        "#b2df8a",  # nvhpc
        "#fdd49e",  # clang
        "#fc8d59",  # clang svml
        "#9ebcda",  # mod2ir
        "#8c96c6",  # mod2ir svml
        "#969696",  # mod2ir jit svml
        "#525252",  # mod2ir jit sleef
    ]

    sns.set_palette(sns.color_palette(colors))
    compilers_comparison_config = """
    {
      "intel": {
        "skylake-avx512": [
          "-O2 -mavx512f -prec-div -fopenmp",
          "-O2 -mavx512f -prec-div -fimf-use-svml -fopenmp"
        ]
      },
      "gcc": {
        "skylake-avx512": [
          "-O3 -march=skylake-avx512 -mtune=skylake -mavx512f -ffast-math -ftree-vectorize -fopenmp",
          "-O3 -march=skylake-avx512 -mtune=skylake -mavx512f -ffast-math -ftree-vectorize -mveclibabi=svml -fopenmp"
        ]
      },
      "nvhpc": {
        "skylake-avx512": [
          "-fast -O3 -mp=autopar -tp=skylake -Msafeptr=all -Minfo -Mvect=simd:512,gather -mavx512vbmi -mavx512vbmi2 -mavx512vl"
        ]
      },
      "clang": {
        "skylake-avx512": [
          "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp",
          "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp -fveclib=SVML",
          "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp jit SVML",
          "-O3 -march=skylake-avx512 -mtune=skylake -ffast-math -fopenmp jit SLEEF"
        ]
      },
      "nmodl_jit": {
        "skylake-avx512": [
          "SVML_nnancontractafn",
          "SLEEF_nnancontractafn"
        ]
      }
    }
    """
    # reference
    hh_expsyn_cpu_reference = load_pickle_result_file(
        [
            "./reference_data/hh_expsyn_mavx512f.pickle",
            "./reference_data/hh_expsyn_nvhpc_cpu.pickle",
        ],
        {},
    )
    json_object = json.dumps(hh_expsyn_cpu_results, indent=4)
    generate_graph_pandas_combined_relative_log(
        hh_expsyn_cpu_reference,
        compilers_comparison_config,
        "reference_hh_expsyn_cpu_relative_log",
        "graphs_output_pandas",
        False,
        xaxis_label="skylake-avx512 Target Microarchitecture",
        plot_size=(10, 3.5),
    )
    # newly collected data
    hh_expsyn_cpu_results = load_pickle_result_file(
        [
            "./hh_expsyn_cpu/benchmark_results.pickle"
        ],
        {},
    )
    json_object = json.dumps(hh_expsyn_cpu_results, indent=4)
    generate_graph_pandas_combined_relative_log(
        hh_expsyn_cpu_results,
        compilers_comparison_config,
        "hh_expsyn_cpu_relative_log",
        "graphs_output_pandas",
        False,
        xaxis_label="skylake-avx512 Target Microarchitecture",
        plot_size=(10, 3.5),
    )


def main():
    plot_cpu_results()


if __name__ == "__main__":
    main()

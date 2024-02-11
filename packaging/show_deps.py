#!/usr/bin/env python3
"""
Small script that can parse TOML files without any dependencies
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

if sys.version_info >= (3, 11):
    # use the standard library module
    from tomllib import load

else:
    import pip

    _earliest_pip_version = (21, 2, 0)
    _current_pip_version = tuple(int(_) for _ in pip.__version__.split("."))
    if _current_pip_version >= _earliest_pip_version:
        # use the vendored version in pip
        from pip._vendor.tomli import load

    else:
        # as a last resort, try to use tomli if it's installed
        try:
            from tomli import load
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"pip version {pip.__version__} found, "
                f"{'.'.join(map(str,_earliest_pip_version))} or above "
                f"or the `tomli` package required\n"
                "You can either install a newer version of pip (using `pip install -U pip`), "
                "or install the `tomli` package (using `pip install tomli`)"
            ) from err


def main():
    """
    Main module
    """
    parser = ArgumentParser()
    parser.add_argument(
        "dir",
        help="the directory containing the `pyproject.toml` file",
    )
    parser.add_argument(
        "--runtime",
        "-r",
        action="store_true",
        help="show only runtime dependencies",
    )
    parser.add_argument(
        "--build",
        "-b",
        action="store_true",
        help="show only build dependencies",
    )
    parser.add_argument(
        "--type",
        "-t",
        help="the type of optional dependency to show (default: show packages from all types)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="list all of the optional dependency types. "
        "Note that this overrides all other options",
    )
    args = parser.parse_args()

    if not (Path(args.dir) / "pyproject.toml").exists():
        raise FileNotFoundError(f"No `pyproject.toml` file found in {args.dir}")

    with open(Path(args.dir) / "pyproject.toml", "rb") as f:
        content = load(f)

    top_key, opt_key = "project", "optional-dependencies"

    if args.list:
        for key in content[top_key].get(opt_key):
            print(key)
        return

    if args.build:
        print("# build time dependencies")
        for package in content["build-system"]["requires"]:
            print(package)

    if args.runtime:
        if content[top_key].get("dependencies"):
            print("# runtime dependencies")
            for package in content[top_key].get("dependencies"):
                print(package)

    if args.type:
        if args.type not in content[top_key].get(opt_key):
            raise ValueError(
                f"The value {args.type} is not a type of optional dependency or `build-system`",
            )
        print(f"# dependencies for {args.type}")
        for item in content[top_key][opt_key][args.type]:
            print(item)

    elif content[top_key].get(opt_key):
        for key in content[top_key].get(opt_key):
            print(f"# dependencies for {key}")
            for item in content[top_key][opt_key][key]:
                print(item)


if __name__ == "__main__":
    main()

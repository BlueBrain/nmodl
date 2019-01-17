## NMODL

This is a source-to-source code generation framework for NMODL.


#### Cloning Source

```
git clone ssh://bbpcode.epfl.ch:22/incubator/nocmodl
```

#### Dependencies

- flex (>=2.6)
- bison (>=3.0)
- CMake (>=3.1)
- C++ compiler (with c++11 support)
- Python3 (>=3.6)
- Python yaml (pyyaml)
- Jinja2 (>=2.10)
- Python textwrap
- pybind11 (which should be fetched in its submodule in ext/pybind11)

#### Getting Dependencies

Many systems have older version of Flex and Bison. Make sure to have latest version of Flex (>=2.6) and Bison (>=3.0).


On macos X we typically install packages via brew/macport and pip :

```
brew install flex bison
pip install pyyaml

# Or
pip3 install pyyaml
```

Make sure to have latest flex/bison in $PATH :


```
export PATH=/usr/local/opt/flex:/usr/local/opt/bison:$PATH
```

On Ubuntu (>=16.04) you should already have recent version of flex/bison:

```
$ flex --version
flex 2.6.4

$ bison --version
bison (GNU Bison) 3.0.4
```

NMODL depends on Python 3, so make sure you have an up-to-date Python installation. On macos X Python 3 can be installed 
through e.g. homebrew. On Ubuntu, depending on your version, Python 3 is either already available by default or can be easily
obtained through

```
$ apt-get install python3
```

Python yaml can be installed as :

```
apt-get install python-yaml
```


#### Build

Once all dependencies are in place, you can build cmake project as :

```
mkdir nocmodl/build
cd nocmodl/build
cmake ..
make
```

If flex / bison is installed in non-standard location then you can do :

```
cmake .. -DCMAKE_PREFIX_PATH="/usr/local/opt/bison/;/usr/local/opt/flex"

 # Or,

cmake .. -DFLEX_EXECUTABLE=/usr/local/opt/flex/bin/flex -DBISON_EXECUTABLE=/usr/local/opt/bison/bin/bison
```

On BB5, you can do:

```
module load cmake/3.12.0 bison/3.0.5 flex/2.6.3 gcc/6.4.0

mkdir build && cd build
cmake ..
make VERBOSE=1
make test
```

#### Running CoreNEURON

You can use NMODL code generator instead of MOD2 for CoreNEURON. You have to simply use extra CMake argument `-DMOD2C` pointing to `nmodl` binary:

```
git clone --recursive https://github.com/BlueBrain/CoreNeuron.git coreneuron
mkdir coreneuron/build && cd coreneuron/build
cmake .. -DMOD2C=/path/nocmodl/build/bin/nmodl
make
make test
```

#### Using NMODL

To run code generator, you can do:

```
./bin/nmodl ../test/input/channel.mod
```

You can independently run lexer, parser or visitors as:

```
./bin/nmodl_lexer --file ../test/input/channel.mod
./bin/nmodl_parser --file ../test/input/channel.mod
./bin/nmodl_visitor --file ../test/input/channel.mod
```


#### Running Test

 You can run unit tests as:

```
 make test
```

 Or individual test binaries with verbode output:

 ```
 ./bin/test/testlexer -s
 ./bin/test/testmodtoken -s
 ./bin/test/testprinter -s
 ./bin/test/testsymtab -s
 ./bin/test/testvisitor -s
 ```


#### Memory Leaks and Clang Tidy

Test memory leaks using :

```
valgrind --leak-check=full --track-origins=yes  ./bin/nmodl_lexer
```

Or using CTest as:

```
ctest -T memcheck
```

If you want to enable `clang-tidy` checks with CMake, make sure to have `CMake >= 3.5` and use following cmake option:

```
cmake .. -DENABLE_CLANG_TIDY=ON
```

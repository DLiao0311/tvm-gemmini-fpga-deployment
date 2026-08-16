# Software setup and clean-machine reproduction

This guide builds the host-side TVM compiler and prepares the Python and RISC-V Linux
cross-compilation environment used by this deployment repository.

## Validation status

- The original development workflow was recorded on Ubuntu 22.04.
- A clean reproduction on Ubuntu 20.04 is currently in progress.
- The TVM fork, branch, and build configuration below reflect the validated project sources.
- Until the Ubuntu 20.04 run reaches Step 3 successfully, Ubuntu 20.04 should be treated as a
  reproduction target rather than a fully validated environment.

Record the Python, CMake, LLVM, compiler, and package versions produced during the clean run.
They will become the pinned environment after validation.

## 1. Install system dependencies

The workflow requires:

- Python 3 with `venv` and development headers;
- CMake and a C/C++ build toolchain;
- LLVM with development headers for TVM code generation;
- the RISC-V 64-bit Linux cross-compiler for Step 3.

On Ubuntu, install the base packages with:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  python3-dev \
  python3-pip \
  python3-venv \
  gcc-riscv64-linux-gnu \
  g++-riscv64-linux-gnu
```

The original environment used LLVM 14. Package availability and installation sources differ
between Ubuntu 22.04 and Ubuntu 20.04, so the LLVM 14 installation method for Ubuntu 20.04 is
still being verified. Before configuring TVM, record the selected installation:

```bash
llvm-config --version
llvm-config --cmakedir
```

If `llvm-config` is versioned, use the corresponding command, such as `llvm-config-14`.

Vivado and Chipyard are not required for Steps 1–3. They are required only when regenerating
the FPGA hardware or bitstream.

## 2. Clone the tested TVM fork

```bash
cd ~
git clone --recursive \
  --branch pr-13770 \
  https://github.com/DLiao0311/tvm.git
cd tvm
git submodule update --init --recursive
```

Confirm the source identity:

```bash
git branch --show-current
git rev-parse HEAD
git submodule status
```

The deployment documentation currently pins:

```text
repository: https://github.com/DLiao0311/tvm
branch:     pr-13770
commit:     463f41dff1e8aacf40267d7d11929236fec114f3
```

The branch already contains the project-specific Gemmini changes. Do not fetch Apache TVM PR
13770 into a new local branch or manually replace `intrin.py` and `pattern_table.py`.

## 3. Create the Python virtual environment

```bash
cd ~
python3 -m venv tvm_venv
source ~/tvm_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Clone this deployment repository if it is not already present, then install its Python
dependencies:

```bash
cd ~
git clone https://github.com/DLiao0311/tvm-gemmini-fpga-deployment.git
cd tvm-gemmini-fpga-deployment
python -m pip install -r requirements.txt
```

This TVM revision requires NumPy 1.x; `requirements.txt` constrains it to `numpy<2`.

## 4. Configure and build host TVM

Create a build directory from a fresh copy of TVM's example configuration:

```bash
cd ~/tvm
mkdir -p build_host
cp cmake/config.cmake build_host/config.cmake
```

Set at least the following entries in `build_host/config.cmake`:

```cmake
set(USE_LLVM ON)
set(USE_AOT_EXECUTOR ON)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_MICRO ON)
set(USE_GEMMINI ON)
set(BUILD_STATIC_RUNTIME OFF)
```

If CMake cannot locate LLVM automatically, replace `ON` with the path to the selected
`llvm-config` executable, for example:

```cmake
set(USE_LLVM /usr/bin/llvm-config-14)
```

Build TVM:

```bash
cmake -S ~/tvm -B ~/tvm/build_host
cmake --build ~/tvm/build_host --parallel "$(nproc)"
```

A successful host build produces:

```text
~/tvm/build_host/libtvm.so
~/tvm/build_host/libtvm_runtime.so
```

## 5. Install the fork's Python package

Keep the virtual environment active and install the Python package in editable mode:

```bash
source ~/tvm_venv/bin/activate
python -m pip uninstall -y apache-tvm tvm
python -m pip install -e ~/tvm/python
```

Do not edit `~/tvm_venv/bin/activate`. Store machine-specific paths in the deployment
repository's ignored local environment file instead:

```bash
cd ~/tvm-gemmini-fpga-deployment
cp configs/environment.example configs/environment.local
```

Edit `configs/environment.local`, set `TVM_HOME` to the new checkout, and load it:

```bash
source configs/environment.local
```

Confirm that Python loads the fork and the matching shared library:

```bash
python3 -c 'import tvm; print(tvm.__file__); print(tvm.base._LIB)' 
```

The Python module path should point into `~/tvm/python/tvm`, and the loaded library should point
into `~/tvm/build_host`.

## 6. Smoke-test the deployment flow

Start with the checked-in scene 1 INT8 model, so dataset preparation and Step 1 are not required
for the first compiler test:

```bash
cd ~/tvm-gemmini-fpga-deployment
source ~/tvm_venv/bin/activate
source configs/environment.local

python3 step2_tvm_compile/export_mlf_with_lut.py \
  --model models/onnx_model/finetuned_onnx_model/finetune_scene1/int8_percentile99_999_symmetric.onnx
```

Expected Step 2 artifacts include:

```text
generated/mlf-int8/mlf.tar
generated/mlf-int8/include/quant_lut.h
generated/mlf-int8/preprocessing.json
generated/mlf-int8/runs/<timestamp>/
```

Then cross-compile the latest LUT-enabled run:

```bash
step3_cross_compile/build_lut_elf.sh
file generated/elf/*.elf
```

The output must be a statically linked RISC-V 64-bit Linux ELF. QEMU cannot execute the
Gemmini RoCC instructions. Runtime validation requires either a matching Spike Gemmini extension
and proxy kernel or the matching Rocket + Gemmini FPGA Linux system.

## 7. Record the Ubuntu 20.04 result

For the current clean-machine reproduction, record:

```bash
lsb_release -ds
python3 --version
cmake --version
llvm-config --version
riscv64-linux-gnu-gcc --version
python -m pip freeze
git -C ~/tvm rev-parse HEAD
```

The Ubuntu 20.04 environment should be called validated only after:

1. `libtvm.so` and `libtvm_runtime.so` build successfully;
2. Python imports the intended fork and library;
3. Step 2 exports the LUT, Relay dumps, and MLF;
4. Step 3 produces a RISC-V Linux ELF;
5. the ELF executes with a matching Gemmini configuration on Spike or FPGA.

## Source note

This guide consolidates the project's original “TVM x Gemmini 自動化建置” development notes
and updates them for the current forked-TVM and vendored-header repository structure. Commands
whose Ubuntu 20.04 behavior has not yet been confirmed are explicitly marked as in progress.

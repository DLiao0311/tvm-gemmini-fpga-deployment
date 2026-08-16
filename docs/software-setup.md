# Software setup and clean-machine reproduction

This guide reproduces the software deployment flow on a clean Ubuntu 20.04 machine:

```text
deployment repository
        -> Python environment
        -> TVM–Gemmini host compiler
        -> MLF export
        -> RISC-V Linux ELF
```

The original workflow was developed on Ubuntu 22.04. Ubuntu 20.04 reproduction is currently
in progress and should not be described as fully validated until the final checklist succeeds.

Vivado and Chipyard are not required here. They are needed only to regenerate the FPGA hardware
and bitstream.

## 1. Clone this deployment repository

```bash
cd ~
git clone https://github.com/DLiao0311/tvm-gemmini-fpga-deployment.git
cd tvm-gemmini-fpga-deployment
```

If the repository is already present, update it instead:

```bash
cd ~/tvm-gemmini-fpga-deployment
git pull
```

All remaining commands assume these locations:

```text
~/tvm-gemmini-fpga-deployment
~/tvm
~/tvm_venv
```

## 2. Install Ubuntu 20.04 system packages

Install the host build tools, Python environment support, and RISC-V Linux cross-compiler:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  ca-certificates \
  cmake \
  git \
  gnupg \
  python3-dev \
  python3-pip \
  python3-venv \
  wget \
  gcc-riscv64-linux-gnu \
  g++-riscv64-linux-gnu
```

### Install LLVM 14

Ubuntu 20.04 does not provide LLVM 14 in its default package repository. Add the official LLVM
Focal LLVM 14 repository, then install the exact version used by this project:

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null

echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" \
  | sudo tee /etc/apt/sources.list.d/llvm14.list

sudo apt update
sudo apt install -y llvm-14 llvm-14-dev clang-14
```

Verify the installation:

```bash
llvm-config-14 --version
clang-14 --version
command -v llvm-config-14
```

The expected LLVM configuration executable is:

```text
/usr/bin/llvm-config-14
```

## 3. Create the Python virtual environment

```bash
cd ~
python3 -m venv tvm_venv
source ~/tvm_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

cd ~/tvm-gemmini-fpga-deployment
python -m pip install -r requirements.txt
```

The requirements include the Python packages used by TVM, ONNX import and quantization,
ONNX Runtime evaluation, and image preprocessing. NumPy is constrained to version 1.x because
this TVM revision is incompatible with NumPy 2.x.

## 4. Clone the tested TVM fork

```bash
cd ~
git clone --recursive \
  --branch pr-13770 \
  https://github.com/DLiao0311/tvm.git
cd ~/tvm
git checkout 463f41dff1e8aacf40267d7d11929236fec114f3
git submodule update --init --recursive
```

Confirm the source revision:

```bash
git rev-parse HEAD
git submodule status
```

The expected TVM revision is:

```text
repository: https://github.com/DLiao0311/tvm
source branch: pr-13770
commit:     463f41dff1e8aacf40267d7d11929236fec114f3
```

Checking out the pinned commit places Git in detached-HEAD state. That is expected for a
reproduction build and prevents a later branch update from silently changing the compiler.

The fork already contains the required Gemmini changes. Do not fetch Apache TVM PR #13770 into
another branch and do not manually replace `intrin.py` or `pattern_table.py`.

## 5. Configure and build host TVM

Create the host build directory and configuration:

```bash
cd ~/tvm
mkdir -p build_host
cp cmake/config.cmake build_host/config.cmake
```

Set the following entries in `~/tvm/build_host/config.cmake`:

```cmake
set(USE_LLVM /usr/bin/llvm-config-14)
set(USE_AOT_EXECUTOR ON)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_MICRO ON)
set(USE_GEMMINI ON)
set(BUILD_STATIC_RUNTIME OFF)
```

Build the host compiler:

```bash
cmake -S ~/tvm -B ~/tvm/build_host
cmake --build ~/tvm/build_host --parallel "$(nproc)"
```

A successful build produces:

```text
~/tvm/build_host/libtvm.so
~/tvm/build_host/libtvm_runtime.so
```

Confirm both files exist:

```bash
ls -l ~/tvm/build_host/libtvm.so ~/tvm/build_host/libtvm_runtime.so
```

## 6. Install and select the TVM fork

Install the fork's Python package into the active virtual environment:

```bash
source ~/tvm_venv/bin/activate
python -m pip install -e ~/tvm/python
```

Create the machine-local environment configuration:

```bash
cd ~/tvm-gemmini-fpga-deployment
cp configs/environment.example configs/environment.local
```

Edit `configs/environment.local` so that it contains the correct TVM checkout:

```bash
export TVM_HOME=/home/<user>/tvm
export TVM_LIBRARY_PATH="${TVM_HOME}/build_host"
export PYTHONPATH="${TVM_HOME}/python${PYTHONPATH:+:${PYTHONPATH}}"

export DIM=16
export SCRATCHPAD_KB=512
export ACCUMULATOR_KB=256
```

Replace `<user>` with the Ubuntu account name, then load the environment:

```bash
source ~/tvm_venv/bin/activate
cd ~/tvm-gemmini-fpga-deployment
source configs/environment.local
```

Verify that Python loads this TVM source tree and its matching shared library:

```bash
python3 -c 'import tvm; print(tvm.__file__); print(tvm.base._LIB)'
```

The paths must point into:

```text
~/tvm/python/tvm
~/tvm/build_host
```

## 7. Export a test MLF

Use the checked-in scene 1 INT8 model so that the first compiler test does not require a dataset
or Step 1 quantization:

```bash
cd ~/tvm-gemmini-fpga-deployment
source ~/tvm_venv/bin/activate
source configs/environment.local

python3 step2_tvm_compile/export_mlf_with_lut.py \
  --model models/onnx_model/finetuned_onnx_model/finetune_scene1/int8_percentile99_999_symmetric.onnx
```

Expected artifacts:

```text
generated/mlf-int8/mlf.tar
generated/mlf-int8/include/quant_lut.h
generated/mlf-int8/preprocessing.json
generated/mlf-int8/runs/<timestamp>/
```

## 8. Cross-compile the RISC-V Linux ELF

```bash
cd ~/tvm-gemmini-fpga-deployment
step3_cross_compile/build_lut_elf.sh
file generated/elf/*.elf
```

The output must be a statically linked RISC-V 64-bit Linux ELF. QEMU cannot execute Gemmini
RoCC instructions. Runtime validation requires a matching Spike Gemmini extension and proxy
kernel or the matching Rocket + Gemmini FPGA Linux system.

## 9. Record the Ubuntu 20.04 environment

After the clean run, record:

```bash
lsb_release -ds
python3 --version
cmake --version
llvm-config-14 --version
riscv64-linux-gnu-gcc --version
python -m pip freeze
git -C ~/tvm rev-parse HEAD
```

Ubuntu 20.04 becomes a validated environment only after:

1. `libtvm.so` and `libtvm_runtime.so` build successfully;
2. Python imports the intended TVM fork and matching library;
3. Step 2 exports the LUT, Relay dumps, and MLF;
4. Step 3 produces a RISC-V Linux ELF;
5. the ELF executes with a matching Gemmini configuration on Spike or FPGA.

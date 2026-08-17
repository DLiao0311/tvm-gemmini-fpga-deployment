# TVM–Gemmini FPGA Deployment Pipeline

An end-to-end deployment pipeline for converting a quantized high-level CNN model into a
RISC-V executable for a Linux-based Rocket Core + Gemmini FPGA SoC.

This project bridges the representation gap between real-world ONNX/QDQ models and the
pattern-constrained Gemmini backend introduced by Apache TVM PR #13770. A CNN-based
people-counting model is used as the primary case study.

## Project status

This repository is being organized from the implementation developed for the thesis:
“TVM-Based CNN Deployment on RISC-V FPGA SoC with Gemmini Accelerator.”

The original software workflow was developed on Ubuntu 22.04. A clean-machine reproduction on
Ubuntu 20.04 is currently in progress; see [`docs/software-setup.md`](docs/software-setup.md)
for the procedure and current validation boundary.

## Validated environment

- Board: Digilent Genesys 2
- SoC: Rocket Core + Gemmini
- Operating system: Linux
- Chipyard fork: `https://github.com/DLiao0311/chipyard`
- Chipyard branch: `genesys2-gemmini-1.13.0`
- Chipyard commit: `6f3015a5`
- Chipyard base: 1.13.0 (`69eba860a352343e4ac6b6df0f3638a79a86ec78`)
- Gemmini hardware commit: `c8bcc68e`
- FPGA-shell commit: `83f9e9fe`
- TVM fork: `https://github.com/DLiao0311/tvm`
- TVM branch: `pr-13770`
- TVM base commit: `463f41dff`
- Vendored Gemmini RoCC software: `1a1a1c6bd` plus configurable hardware-parameter guards

## Key results

- Gemmini model inference latency: approximately 0.550 s
- End-to-end latency: 10.67 s to 0.897 s
- End-to-end latency reduction: approximately 91.6%
- People-counting MAE: 0.297 people

## Deployment flow

```text
Fine-tuned CNN
      |
      v
ONNX model and INT8 post-training quantization
      |
      v
Relay import, layout and quantized-operator normalization
      |
      v
Gemmini pattern matching and legalization
      |
      v
AOT C code generation and Model Library Format export
      |
      v
Gemmini API integration and RISC-V cross-compilation
      |
      v
Target objects linked into a Linux ELF
      |
      v
Linux on Rocket Core + Gemmini FPGA SoC
```

## Why generate C before the ELF?

Most mature ML deployment toolchains do not expose C source as a deliberate integration
boundary. They lower a model through an internal representation directly to target-specific
object code, then link those objects with an application and runtime. This is effective when
the compiler already has complete support for the target CPU or accelerator, but it provides
little room to inspect or adapt accelerator calls before machine code is emitted. An object
file is already compiled for a particular ISA and ABI; a host object cannot subsequently be
cross-compiled into a RISC-V object.

This project intentionally retains generated C between TVM lowering and ELF creation because
Gemmini integration is part of the research problem, not merely a final packaging step. The C
boundary makes it possible to:

- map supported Relay subgraphs to the Gemmini software API while leaving surrounding
  application and runtime code in conventional C;
- adapt generated calls to the Chipyard/Gemmini API version used by the FPGA design;
- propagate quantization, layout, fused-activation, tiling, and memory-configuration decisions
  into code that can be inspected and debugged;
- compile the same generated model code with the RISC-V toolchain and the exact `DIM`,
  scratchpad, and accumulator configuration implemented in hardware; and
- link model code, parameters, preprocessing, postprocessing, TVM runtime support, and the
  application entry point into one deployable Linux ELF.

Model Library Format (MLF) preserves this boundary by packaging the AOT-generated C sources,
headers, parameters, and metadata for the downstream platform build. Step 2 therefore performs
model-level lowering and exports an inspectable integration artifact; Step 3 performs the
platform-specific cross-compilation and linking:

```text
Quantized ONNX
      -> TVM Relay/TIR and Gemmini legalization
      -> generated C plus MLF metadata and parameters
      -> RISC-V compilation into target-specific objects
      -> link with the runtime and application
      -> Rocket + Gemmini Linux ELF
```

The extra C stage is consequently not a workaround for ELF generation. It is the explicit
hardware/software co-design boundary that allows a real quantized CNN, TVM's pattern-constrained
Gemmini backend, the Gemmini accelerator API, and a concrete FPGA configuration to be validated
together.

## Main contributions

- Built a repeatable ONNX-to-Gemmini compilation and deployment workflow.
- Normalized Relay layouts, quantized operators, and fused activation patterns so that a
  high-level quantized CNN can match the TVM Gemmini backend.
- Adapted Gemmini code generation to the Chipyard 1.13.0 API.
- Fixed fused clip/ReLU activation propagation through Gemmini lowering.
- Integrated AOT, Model Library Format, RISC-V cross-compilation, Linux runtime, and FPGA
  execution.
- Evaluated PTQ strategies, implicit padding, LUT-based preprocessing, and Gemmini hardware
  configurations under limited FPGA resources.

## Upstream foundation

The original Gemmini backend is based on Apache TVM PR #13770. This repository does not
claim authorship of that backend. The work here focuses on preparing real-world quantized CNN
graphs for its supported patterns, adapting generated calls to Chipyard 1.13.0, and completing
the end-to-end deployment and validation flow.

The Genesys 2 FPGA shell and harness are provided by the public
[`stanley-666/chipyard_fpga_genesys2`](https://github.com/stanley-666/chipyard_fpga_genesys2)
project for Chipyard 1.13.0. They remain an external dependency and retain their
BSD-3-Clause and Apache-2.0 license notices. This project's hardware contribution is the
Rocket/Gemmini configuration and its integration into the end-to-end deployment flow; see
[`hardware/README.md`](hardware/README.md).

## Repository layout

```text
step1_quantization/   Post-training quantization and calibration
step2_tvm_compile/    Relay transformations, INT8 input externalization, LUT and MLF export
step3_cross_compile/  RISC-V cross-compilation
runtime/              Application, preprocessing, and postprocessing code
third_party/          Pinned Gemmini software API headers and licenses
configs/              Reproducible software environment configuration
hardware/             Genesys 2 dependency, FPGA flow, and hardware metadata
docs/                 Architecture, modifications, deployment, and limitations
generated/            Local generated MLF, Relay, and ELF artifacts (not tracked)
```

Generated artifacts follow the workflow stages:

```text
models/<fp32-model>.onnx
        -> Step 1 -> generated/quantized_models/<model>/
        -> Step 2 -> generated/mlf-int8/
        -> Step 3 -> generated/elf/
```

## Basic usage

Install Python dependencies, then install the tested TVM fork in editable mode as described in
the [software setup guide](docs/software-setup.md). Configure the local toolchain paths:

```bash
cp configs/environment.example configs/environment.local
# Edit configs/environment.local, then:
source configs/environment.local
```

### Step 1: Quantize the model

```bash
python step1_quantization/quantize_models.py \
  --dataset /path/to/images \
  /path/to/people-counting-fp32.onnx
```

### Step 2: Export MLF

Without LUT:

```bash
python step2_tvm_compile/export_mlf.py --model /path/to/people-counting-int8.onnx
```

With LUT:

```bash
python step2_tvm_compile/export_mlf_with_lut.py \
  --model /path/to/people-counting-int8.onnx
```

### Step 3: Cross-compile the ELF

Without LUT:

```bash
step3_cross_compile/build_elf.sh
```

With LUT:

```bash
step3_cross_compile/build_lut_elf.sh
```

The ELF is written to `generated/elf/`.

## Scope and limitations

- The workflow is validated primarily with the people-counting CNN used in the thesis.
- It supports the layouts, operators, and quantization patterns required by that model.
- It does not claim automatic deployment of arbitrary CNN architectures.
- Full FPGA reproduction requires a compatible Chipyard and Genesys 2 environment.
- Genesys 2 board support is an external, attributed dependency; it is not authored by this
  project.

## License

Original code and documentation developed for this repository are released under the
[Apache License 2.0](LICENSE), unless otherwise noted.

Third-party components retain their original licenses and copyright notices:

- [Apache TVM](https://github.com/apache/tvm): Apache License 2.0. TVM source modifications
  are maintained in the separate TVM fork and remain subject to TVM's license and notices.
- Gemmini RoCC software headers: see
  [`third_party/gemmini-rocc-tests/LICENSE`](third_party/gemmini-rocc-tests/LICENSE) and
  [`SOURCE.md`](third_party/gemmini-rocc-tests/SOURCE.md).
- [Genesys 2 board support](https://github.com/stanley-666/chipyard_fpga_genesys2):
  BSD-3-Clause and Apache-2.0 (`LICENSE.SiFive`). Its source is referenced as an external
  dependency and is not currently redistributed by this repository.

The ONNX models in `models/onnx_model/` are authored by the project owner and intentionally
published as reproducibility artifacts. Datasets, images, ground-truth annotations, generated
artifacts, and FPGA bitstreams are not covered by the repository's Apache-2.0 statement unless
a file or release explicitly says otherwise.

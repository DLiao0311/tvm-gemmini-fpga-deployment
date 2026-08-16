# TVM–Gemmini FPGA Deployment Pipeline

An end-to-end deployment pipeline for converting a quantized high-level CNN model into a
RISC-V executable for a Linux-based Rocket Core + Gemmini FPGA SoC.

This project bridges the representation gap between real-world ONNX/QDQ models and the
pattern-constrained Gemmini backend introduced by Apache TVM PR #13770. A CNN-based
people-counting model is used as the primary case study.

## Project status

This repository is being organized from the implementation developed for the thesis:
“TVM-Based CNN Deployment on RISC-V FPGA SoC with Gemmini Accelerator.”

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
AOT code generation and Model Library Format export
      |
      v
RISC-V cross-compilation
      |
      v
Linux on Rocket Core + Gemmini FPGA SoC
```

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
its build documentation. Configure the local toolchain paths:

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

Original code and documentation developed for this repository are intended to be released
under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), unless otherwise
noted. A root `LICENSE` file will be added before the repository is made public.

Third-party components retain their original licenses and copyright notices:

- [Apache TVM](https://github.com/apache/tvm): Apache License 2.0. TVM source modifications
  are maintained in the separate TVM fork and remain subject to TVM's license and notices.
- Gemmini RoCC software headers: see
  [`third_party/gemmini-rocc-tests/LICENSE`](third_party/gemmini-rocc-tests/LICENSE) and
  [`SOURCE.md`](third_party/gemmini-rocc-tests/SOURCE.md).
- [Genesys 2 board support](https://github.com/stanley-666/chipyard_fpga_genesys2):
  BSD-3-Clause and Apache-2.0 (`LICENSE.SiFive`). Its source is referenced as an external
  dependency and is not currently redistributed by this repository.

The ONNX models, datasets, images, ground-truth annotations, generated artifacts, and FPGA
bitstreams are not covered by the repository's Apache-2.0 statement unless a file or release
explicitly says otherwise. Their redistribution rights must be confirmed separately before
publication.

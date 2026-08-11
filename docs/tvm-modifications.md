# Gemmini Adaptation Notes

## Goal

Make TVM-generated Gemmini AOT C code match the Gemmini revision used by Chipyard 1.13.0,
preserve fused ReLU-as-clip, and keep the software headers aligned with the selected hardware
configuration.

## Kept Changes

### 1. TVM Gemmini conv call signature update

File:
- `python/tvm/contrib/gemmini/intrin.py` in the TVM fork

What changed:
- Updated `tiled_conv_auto` emission to use:
  - `IN_ROW_DIM`, `IN_COL_DIM`
  - `OUT_ROW_DIM`, `OUT_COL_DIM`
- Updated `tiled_conv_dw_auto` emission the same way.
- Added a guard that raises for non-zero padding because the current Chipyard 1.13.0 path is only validated for zero-padding.

Why:
- Older TVM emission assumed a single square-dimension style interface.
- Chipyard 1.13.0 expects row/col dimensions explicitly.

Observed result:
- Generated `lib1.c` now emits calls such as:
  - `tiled_conv_auto(1, 128, 128, ..., 128, 128, ...)`

### 2. Preserve activation when pattern is `clip(requantize(...))`

File:
- `python/tvm/contrib/gemmini/pattern_table.py` in the TVM fork

What changed:
- When the outer node is `clip`, treat it as activation.
- Propagate activation quantization metadata from the inner `qnn.requantize`:
  - `activation_scale_in`
  - `activation_offset_in`
  - `activation_scale_out`
  - `activation_offset_out`
- Set `self.activation = self.has_activation` instead of hard-coding `False`.

Why:
- `MergeComposite` already matched patterns like:
  - `qnn.conv2d -> nn.bias_add -> qnn.requantize -> clip`
- But the parser dropped the outer `clip` without marking activation, so generated Gemmini calls used `act=0`.

Observed result:
- ReLU/clip can now propagate into Gemmini lowering, allowing `tiled_conv_auto(..., act, ...)` to become `act=1` when matched.

### 3. Pin the deployment API to the Gemmini revision used by Chipyard 1.13.0

Vendored dependency:
- `third_party/gemmini-rocc-tests` in this deployment repository

What changed:
- Vendored `gemmini-rocc-tests` commit
  `1a1a1c6bd60df6d7cae3d87aac96c8f406cae084`, selected by Gemmini commit
  `25809f78323a729ef76fb68f3cedd8a24da2942b`.
- Retained the upstream license and source metadata alongside the headers.

Why:
- `25809f7` is the Gemmini revision used by the validated Chipyard 1.13.0 checkout.
- The deployment repository vendors the matching Gemmini software API snapshot so Step 3 does
  not depend on an external Chipyard or TVM source tree for headers.

### 4. Allow hardware parameters to be selected during cross-compilation

File:
- `third_party/gemmini-rocc-tests/include/gemmini_params.h` in this deployment repository

What changed:
- Wrapped `DIM`, `BANK_ROWS`, and `ACC_ROWS` defaults with `#ifndef` guards.

```c
#ifndef DIM
#define DIM 16
#endif

#ifndef BANK_ROWS
#define BANK_ROWS 4096
#endif

#ifndef ACC_ROWS
#define ACC_ROWS 1024
#endif
```

Why:
- Step 3 derives row counts from the selected DIM, scratchpad capacity, and accumulator
  capacity, then supplies them through compiler definitions.
- For the current DIM 16, 512 KiB scratchpad, and 256 KiB accumulator configuration, the ELF
  is compiled with `BANK_ROWS=8192` and `ACC_ROWS=4096`.
- Without the guards, the header overwrites those values with its built-in defaults.

## Software headers and Spike configuration

These are related but separate configurations:

- The ELF uses the pinned `third_party/gemmini-rocc-tests/include/gemmini_params.h` snapshot in
  this deployment repository.
- Spike uses the `gemmini_params.h` compiled into its `libgemmini.so` extension.

Both sides must use matching DIM, scratchpad, and accumulator parameters. The validated Spike
extension is configured for DIM 16, 512 KiB scratchpad, and 256 KiB accumulator.

## Validation Checklist

After re-running the ONNX-to-AOT pipeline:

1. Check Relay after Gemmini legalization:
   - `generated/mlf/07_after_gemmini_legalize.relay` when IR dumping is enabled
   - Confirm Gemmini conv ops carry activation info.

2. Check generated AOT C:
   - `generated/mlf/runs/<timestamp>/codegen/host/src/default_lib1.c`
   - Confirm:
     - `tiled_conv_auto` uses row/col dimensions
     - the `act` argument is `1` where clip/ReLU should be fused

3. Cross-compile with the vendored Gemmini headers:
   - Run `step3_cross_compile/build_elf.sh` or `build_lut_elf.sh`.
   - Confirm the reported `BANK_ROWS` and `ACC_ROWS` match the target configuration.

4. Run the ELF with a matching FPGA design or Spike Gemmini extension.

## Notes

- Clone the tested TVM fork with recursive submodules and build `build_host/libtvm.so` from that
  checkout. An upstream or prebuilt TVM installation does not contain the required Gemmini
  changes.
- Configure the Python environment to import the same TVM checkout and set `TVM_LIBRARY_PATH`
  to its `build_host` directory.
- After that initial build, edits limited to `intrin.py` or `pattern_table.py` only require
  re-running the export pipeline; rebuilding `libtvm.so` is not required for Python-only edits.
- Changes to TVM C++ sources still require rebuilding the TVM libraries.
- Changes to the vendored Gemmini software headers take effect when Step 3 recompiles the
  RISC-V ELF.

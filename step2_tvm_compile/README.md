# Step 2: TVM–Gemmini compilation

Convert a quantized ONNX model into a Gemmini Model Library Format (MLF) package.

- `export_mlf.py`: export MLF without an RGB preprocessing LUT.
- `export_mlf_with_lut.py`: export an INT8-input MLF and generate the matching RGB preprocessing LUT.

## Environment

```bash
source ~/tvm_venv/bin/activate
```

Clone the tested TVM–Gemmini fork with recursive submodules and build
`build_host/libtvm.so` from that checkout. Configure the Python environment to import the same
checkout and provide its `build_host` directory through `TVM_LIBRARY_PATH`. An upstream or
prebuilt TVM installation does not contain the required Gemmini changes. This TVM revision
requires NumPy 1.x.

## Run without LUT

From the repository root:

```bash
python3 step2_tvm_compile/export_mlf.py \
  --model generated/quantized_models/<source-model-name>/int8_percentile_99_9_symmetric.onnx
```

## Run with LUT

From the repository root:

```bash
python3 step2_tvm_compile/export_mlf_with_lut.py \
  --model generated/quantized_models/<source-model-name>/int8_percentile_99_9_symmetric.onnx
```

If the model uses different RGB normalization parameters:

```bash
python3 step2_tvm_compile/export_mlf_with_lut.py \
  --model /path/to/int8-model.onnx \
  --mean 0.485 0.456 0.406 \
  --std 0.229 0.224 0.225
```

## Output

Outputs are written under `generated/mlf-int8/`:

```text
include/quant_lut.h
preprocessing.json
mlf.tar
runs/<timestamp>/
```

The directory also contains Relay IR dumps for inspecting each transformation stage. Use
`--output /path/to/output` to select another output directory.

# Step 1: Post-training quantization

Provide the image directory explicitly with `--dataset`. It must contain at least 240 images.
The script uses a fixed random seed to split them into 180 calibration images and 60 test images.
The two sets never overlap, and nested image directories are supported.

For each FP32 ONNX model, the workflow generates:

- symmetric INT8 using MinMax calibration;
- symmetric INT8 using Percentile calibration (99.9 by default); and
- `evaluation.csv` containing output differences against the FP32 model on all 60 test images.

This deployment workflow intentionally uses a fixed model contract:

- input: NCHW float32, `1 x 3 x 512 x 512`;
- output: NCHW float32, `1 x 1 x 128 x 128`.

Dynamic ONNX dimensions are materialized to these static dimensions before quantization. A model
with incompatible fixed dimensions or runtime output is rejected.

```bash
source ~/tvm_venv/bin/activate

python step1_quantization/quantize_models.py \
  --dataset /path/to/images \
  /path/to/people-counting-fp32.onnx
```

Outputs are written under `generated/quantized_models/<source-model-name>/`:

```text
int8_minmax_symmetric.onnx
int8_percentile_99_9_symmetric.onnx
evaluation.csv
quantization.json
```

The selected image directory is not copied into this repository. The complete `generated/`
directory is ignored by Git.

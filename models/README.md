# Models

The ONNX models under `models/onnx_model/` are authored by the project owner and intentionally
published as reference inputs for reproducing the quantization and TVM compilation workflow.
Their inclusion does not grant rights to any training images or third-party datasets, which are
not redistributed by this repository.

## Layout

```text
onnx_model/
├── origin_trained_model/       Original trained FP32 and selected INT8 models
└── finetuned_onnx_model/
    ├── Model_A_without_finetune.onnx
    ├── finetune_scene1/        Scene 1 FP32 and PTQ variants
    ├── finetune_scene2/        Scene 2 FP32 and PTQ variants
    └── finetune_scene3/        Scene 3 FP32 and PTQ variants
```

Each scene directory contains the fine-tuned FP32 model and MinMax/Percentile INT8 variants.
Both symmetric and asymmetric historical quantization outputs are archived. The deployment
pipeline documented in this repository uses the symmetric model whose calibration method is
selected for the experiment.

File hashes are recorded in [`SHA256SUMS`](SHA256SUMS). Verify all published models from the
repository root with:

```bash
sha256sum --check models/SHA256SUMS
```

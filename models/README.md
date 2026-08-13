# Models

The ONNX models under `models/onnx_model/` are tracked as reference inputs for reproducing
the quantization and TVM compilation workflow. Other model binaries remain local unless they
are explicitly added to the repository.

Models used by the original research workspace:

| Model | SHA-256 |
|---|---|
| `people_counting.onnx` | `37cd2ef3c7933551a3c0b8d36bff78f805f08cd58e957efca390b9e5becfe7c3` |
| `shin_crowd_model_float32.onnx` | `c6e23889df56d36f505711441f48a50cb17cd943a5a5fd22d4419394ba6424ab` |
| `shin_crowd_model_int8_percentile_symmetric.onnx` | `3e8d5436e88e9b2b80fdd3ac78521d71bf966050929b75bb043891258e94442e` |

These files are small enough to use regular Git and do not require Git LFS. Their inclusion in
the repository does not imply that the training datasets are redistributed or covered by the
repository's Apache-2.0 license. Confirm the model and training-data redistribution rights
before making the repository public.

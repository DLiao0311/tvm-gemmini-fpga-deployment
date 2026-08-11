# Runtime

The runtime loads a 512 x 512 RGB image, runs the generated TVM model, and reports
latency, throughput, and the density-map sum.

- `run.c`: float-input model without LUT preprocessing.
- `run_with_lut.c`: INT8-input model with the generated RGB quantization LUT.

Step 3 selects the matching runtime automatically when building the ELF.

## Single inference

Without LUT:

```bash
generated/elf/pc_gemmini_dim16_sp512kb_acc256kb_mlf.elf image.jpg
```

With LUT:

```bash
generated/elf/pc_gemmini_dim16_sp512kb_acc256kb_throughput_lut_mlf_o3.elf image.jpg
```

## Benchmark

Add the warm-up and iteration counts:

```bash
generated/elf/pc_gemmini_dim16_sp512kb_acc256kb_throughput_lut_mlf_o3.elf \
  image.jpg \
  --warmup 5 \
  --iterations 20
```

Input images are loaded as RGB and resized to 512 x 512 with bilinear interpolation. The
reported latency includes normalization or LUT preprocessing plus model inference; image
decoding and resizing occur before the timed region.

Automatic resizing is provided only for convenient evaluation with the original dataset. The
target camera pipeline is expected to provide 512 x 512 RGB images, in which case resizing is
bypassed. Reported preprocessing and end-to-end latency measurements do not include image
decoding or resizing time.

The density-map values are divided by 1000 before summation to match the model's training
target scale.

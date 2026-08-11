#!/usr/bin/env python3
"""Export Gemmini MLF with an INT8 model input and a generated RGB LUT.

The input model is expected to contain one input-boundary ``qnn.quantize`` after
the NCHW-to-NHWC normalization performed by ``export_mlf.py``. This script:

1. imports the quantized ONNX model;
2. runs the existing layout and symmetric-quantization cleanup passes;
3. replaces ``qnn.quantize(float_input, scale, zero_point)`` with a new INT8
   NHWC function input;
4. generates a 3 x 256 RGB preprocessing/quantization LUT;
5. lowers the rewritten Relay module to Gemmini and exports MLF; and
6. verifies that the exported MLF metadata describes the INT8 input.

The transformation externalizes only input quantization. Internal layer
quantization and requantization remain part of the Relay model.
"""

import argparse
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import onnx
import tvm
from tvm import relay

from export_mlf import (
    build_mlf,
    dump_ir,
    gemmini_lower,
    print_op_stats,
    step1_one_shot_layout_rewrite,
    step2_symmetric_cleanup,
)


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def fixed_onnx_input(onnx_model):
    """Return the single non-initializer ONNX input name and its fixed shape."""
    initializer_names = {item.name for item in onnx_model.graph.initializer}
    inputs = [item for item in onnx_model.graph.input if item.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError(f"expected exactly one model input, found {len(inputs)}")

    model_input = inputs[0]
    shape = []
    for dim in model_input.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError("only fixed-shape ONNX inputs are supported")
        shape.append(int(dim.dim_value))
    if len(shape) != 4:
        raise ValueError(f"expected a four-dimensional NCHW input, found {shape}")
    return model_input.name, tuple(shape)


def load_quantized_onnx(onnx_path):
    """Import a fixed-shape quantized ONNX model into Relay."""
    onnx_model = onnx.load(str(onnx_path))
    input_name, input_shape = fixed_onnx_input(onnx_model)
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape={input_name: input_shape},
        freeze_params=True,
    )
    return relay.transform.InferType()(mod), params


def scalar_constant(expr, label):
    if not isinstance(expr, relay.Constant):
        raise ValueError(f"input {label} must be a Relay Constant")
    array = np.asarray(expr.data.numpy())
    if array.size != 1:
        raise ValueError(f"input {label} must be scalar, found shape {array.shape}")
    return array.reshape(()).item()


class InputQuantizeFinder(relay.ExprVisitor):
    """Find qnn.quantize calls that consume the main function input directly."""

    def __init__(self, input_var):
        super().__init__()
        self.input_var = input_var
        self.matches = []

    def visit_call(self, call):
        if (
            isinstance(call.op, tvm.ir.Op)
            and call.op.name == "qnn.quantize"
            and call.args[0].same_as(self.input_var)
        ):
            self.matches.append(call)
        super().visit_call(call)


class ReplaceInputQuantize(relay.ExprMutator):
    """Replace one selected input-boundary qnn.quantize with an INT8 input."""

    def __init__(self, quantize_call, new_input):
        super().__init__()
        self.quantize_call = quantize_call
        self.new_input = new_input
        self.replacement_count = 0

    def visit_call(self, call):
        if call.same_as(self.quantize_call):
            self.replacement_count += 1
            return self.new_input
        return super().visit_call(call)


def externalize_input_quantization(mod):
    """Replace the sole input qnn.quantize with a typed INT8 function input."""
    mod = relay.transform.InferType()(mod)
    main = mod["main"]
    if len(main.params) != 1:
        raise ValueError(f"expected exactly one Relay input, found {len(main.params)}")

    old_input = main.params[0]
    old_type = old_input.checked_type
    if str(old_type.dtype) != "float32":
        raise ValueError(f"expected float32 input before externalization, found {old_type.dtype}")

    finder = InputQuantizeFinder(old_input)
    finder.visit(main.body)
    if len(finder.matches) != 1:
        raise ValueError(
            "expected exactly one input-boundary qnn.quantize, "
            f"found {len(finder.matches)}"
        )

    quantize_call = finder.matches[0]
    out_dtype = str(quantize_call.attrs.out_dtype)
    if out_dtype not in ("int8", "uint8"):
        raise ValueError(f"unsupported externalized input dtype: {out_dtype}")

    scale = float(scalar_constant(quantize_call.args[1], "scale"))
    zero_point = int(scalar_constant(quantize_call.args[2], "zero point"))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"input scale must be finite and positive, found {scale}")

    shape = tuple(int(dim) for dim in old_type.shape)
    new_input = relay.var(old_input.name_hint, shape=shape, dtype=out_dtype)
    mutator = ReplaceInputQuantize(quantize_call, new_input)
    new_body = mutator.visit(main.body)
    if mutator.replacement_count != 1:
        raise RuntimeError(
            f"expected one qnn.quantize replacement, made {mutator.replacement_count}"
        )

    new_main = relay.Function(
        [new_input],
        new_body,
        ret_type=None,
        type_params=main.type_params,
        attrs=main.attrs,
    )
    new_mod = tvm.IRModule.from_expr(new_main)
    new_mod = relay.transform.InferType()(new_mod)

    rewritten_input = new_mod["main"].params[0].checked_type
    if str(rewritten_input.dtype) != out_dtype:
        raise RuntimeError("Relay type inference did not preserve the externalized input dtype")

    return new_mod, {
        "scale": scale,
        "zero_point": zero_point,
        "dtype": out_dtype,
        "shape": shape,
        "axis": int(getattr(quantize_call.attrs, "axis", -1)),
    }


def round_away_from_zero(value):
    """Match C roundf behavior used by the generated TVM quantization loop."""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def quantized_pixel(pixel, channel, mean, std, scale, zero_point, dtype):
    normalized = (pixel / 255.0 - mean[channel]) / std[channel]
    value = round_away_from_zero(normalized / scale) + zero_point
    limits = {"int8": (-128, 127), "uint8": (0, 255)}
    qmin, qmax = limits[dtype]
    return max(qmin, min(qmax, value))


def build_lut(mean, std, quantization):
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("RGB LUT generation requires exactly three mean and std values")
    if any(not math.isfinite(value) for value in (*mean, *std)):
        raise ValueError("mean and std values must be finite")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be positive")

    return [
        [
            quantized_pixel(
                pixel,
                channel,
                mean,
                std,
                quantization["scale"],
                quantization["zero_point"],
                quantization["dtype"],
            )
            for pixel in range(256)
        ]
        for channel in range(3)
    ]


def write_lut_header(path, lut, mean, std, quantization):
    path.parent.mkdir(parents=True, exist_ok=True)
    c_type = "int8_t" if quantization["dtype"] == "int8" else "uint8_t"
    lines = [
        "/* Generated by export_mlf_with_lut.py. */",
        "/* Do not edit: regenerate this file when model quantization changes. */",
        "#ifndef TVM_GEMMINI_GENERATED_QUANT_LUT_H_",
        "#define TVM_GEMMINI_GENERATED_QUANT_LUT_H_",
        "",
        "#include <stdint.h>",
        "",
        f"/* input_scale={quantization['scale']:.9g}, "
        f"input_zero_point={quantization['zero_point']}, dtype={quantization['dtype']} */",
        f"/* RGB mean={list(mean)}, std={list(std)} */",
        f"static const {c_type} quant_lut[3][256] = {{",
    ]
    for channel, values in enumerate(lut):
        lines.append(f"    /* channel {channel} */")
        lines.append("    {")
        for offset in range(0, 256, 16):
            chunk = ", ".join(f"{value:4d}" for value in values[offset : offset + 16])
            suffix = "," if offset + 16 < 256 else ""
            lines.append(f"        {chunk}{suffix}")
        lines.append("    }," if channel < 2 else "    }")
    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path, mean, std, quantization, lut_header):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "channel_order": "RGB",
        "input_layout": "NHWC",
        "pixel_range": [0, 255],
        "mean": list(mean),
        "std": list(std),
        "input_scale": quantization["scale"],
        "input_zero_point": quantization["zero_point"],
        "input_dtype": quantization["dtype"],
        "input_shape": list(quantization["shape"]),
        "quantization_axis": quantization["axis"],
        "lut_header": str(lut_header),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_mlf_input(mlf_tar, quantization):
    with tarfile.open(mlf_tar, "r") as archive:
        member = next(
            (item for item in archive.getmembers() if item.name.endswith("metadata.json")),
            None,
        )
        if member is None:
            raise RuntimeError("MLF archive does not contain metadata.json")
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("failed to read MLF metadata.json")
        metadata = json.load(stream)

    module = metadata["modules"]["default"]
    function = module["memory"]["functions"]["main"][0]
    inputs = function["inputs"]
    if len(inputs) != 1:
        raise RuntimeError(f"expected one MLF input, found {len(inputs)}")
    input_info = next(iter(inputs.values()))

    expected_size = math.prod(quantization["shape"])
    if quantization["dtype"] != "int8":
        expected_size *= 1
    actual_dtype = input_info["dtype"]
    actual_size = int(input_info["size"])
    if actual_dtype != quantization["dtype"] or actual_size != expected_size:
        raise RuntimeError(
            "MLF input contract mismatch: "
            f"expected {quantization['dtype']} / {expected_size} bytes, "
            f"found {actual_dtype} / {actual_size} bytes"
        )
    print(f"OK: MLF input is {actual_dtype}, {actual_size} bytes")


def parse_args():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Externalize input quantization and export INT8-input Gemmini MLF"
    )
    parser.add_argument("--model", required=True, type=Path, help="Quantized ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "generated" / "mlf-int8",
        help="Output directory (default: generated/mlf-int8)",
    )
    parser.add_argument("--mean", nargs=3, type=float, default=DEFAULT_MEAN, metavar=("R", "G", "B"))
    parser.add_argument("--std", nargs=3, type=float, default=DEFAULT_STD, metavar=("R", "G", "B"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mod, params = load_quantized_onnx(args.model.resolve())
    dump_ir(mod, args.output / "00_from_onnx.relay")

    mod = step1_one_shot_layout_rewrite(mod)
    mod = step2_symmetric_cleanup(mod, args.output / "05_after_layout_fuse.relay")
    mod, quantization = externalize_input_quantization(mod)
    dump_ir(mod, args.output / "06_after_externalize_input_quantization.relay")
    print(
        "OK: externalized input quantization: "
        f"shape={quantization['shape']}, dtype={quantization['dtype']}, "
        f"scale={quantization['scale']:.9g}, zero_point={quantization['zero_point']}"
    )

    lut = build_lut(tuple(args.mean), tuple(args.std), quantization)
    lut_header = args.output / "include" / "quant_lut.h"
    write_lut_header(lut_header, lut, tuple(args.mean), tuple(args.std), quantization)
    write_manifest(
        args.output / "preprocessing.json",
        tuple(args.mean),
        tuple(args.std),
        quantization,
        lut_header.relative_to(args.output),
    )
    print(f"OK: generated {lut_header}")

    print_op_stats(mod, "After input quantization externalization")
    mod = gemmini_lower(mod)
    dump_ir(mod, args.output / "07_after_gemmini_legalize.relay")
    build_mlf(mod, params, output_dir=str(args.output))
    validate_mlf_input(args.output / "mlf.tar", quantization)


if __name__ == "__main__":
    main()

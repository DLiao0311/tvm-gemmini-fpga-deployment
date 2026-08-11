#!/usr/bin/env python3
"""Generate MinMax and Percentile symmetric INT8 ONNX models.

The dataset is deterministically split into 180 calibration images and 60 test
images by default. Test images are never presented to the quantizer.
"""

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 1, 128, 128)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create MinMax and Percentile symmetric INT8 ONNX models"
    )
    parser.add_argument("models", nargs="+", type=Path, help="FP32 ONNX model(s)")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory containing calibration and test candidate images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "generated" / "quantized_models",
        help="Root directory for per-model quantization outputs",
    )
    parser.add_argument("--calibration-count", type=int, default=180)
    parser.add_argument("--test-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile calibration threshold (default: 99.9)",
    )
    return parser.parse_args()


def collect_images(dataset_dir):
    images = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise ValueError(f"no supported images found in {dataset_dir}")
    return images


def split_dataset(images, calibration_count, test_count, seed):
    required = calibration_count + test_count
    if calibration_count <= 0 or test_count <= 0:
        raise ValueError("calibration-count and test-count must both be positive")
    if len(images) < required:
        raise ValueError(f"dataset needs at least {required} images, found {len(images)}")

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    calibration = shuffled[:calibration_count]
    test = shuffled[calibration_count:required]
    if set(calibration) & set(test):
        raise RuntimeError("calibration and test splits overlap")
    return calibration, test


def model_input_spec(model_path):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("this workflow currently supports one model input")
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if len(shape) != 4:
        raise ValueError(f"expected a four-dimensional NCHW input, found {shape}")
    for declared, required, label in zip(shape, INPUT_SHAPE, ("batch", "channel", "height", "width")):
        if isinstance(declared, int) and declared != required:
            raise ValueError(
                f"model {label} dimension is {declared}, but this workflow requires {required}"
            )
    if model_input.type != "tensor(float)":
        raise ValueError(f"expected a float32 model input, found {model_input.type}")
    return model_input.name, INPUT_SHAPE[2], INPUT_SHAPE[3]


def set_tensor_shape(value_info, shape):
    dimensions = value_info.type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError(
            f"cannot set {value_info.name} to shape {shape}: model rank is {len(dimensions)}"
        )
    for dimension, value in zip(dimensions, shape):
        dimension.ClearField("dim_param")
        dimension.dim_value = value


def write_static_model(source_path, destination_path):
    """Materialize the deployment input/output contract in a temporary FP32 ONNX model."""
    model = onnx.load(str(source_path))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("this workflow requires exactly one model input and one model output")
    set_tensor_shape(model.graph.input[0], INPUT_SHAPE)
    set_tensor_shape(model.graph.output[0], OUTPUT_SHAPE)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination_path))


def preprocess(image_path, height, width):
    image = Image.open(image_path).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    normalized = (array - DEFAULT_MEAN) / DEFAULT_STD
    return np.expand_dims(normalized, axis=0)


class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name, height, width):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.height = height
        self.width = width
        self.rewind()

    def get_next(self):
        try:
            image_path = next(self.iterator)
        except StopIteration:
            return None
        return {self.input_name: preprocess(image_path, self.height, self.width)}

    def rewind(self):
        self.iterator = iter(self.image_paths)


def output_paths(model_path, output_dir, percentile):
    percentile_label = str(percentile).replace(".", "_")
    return {
        "minmax": output_dir / "int8_minmax_symmetric.onnx",
        "percentile": output_dir / f"int8_percentile_{percentile_label}_symmetric.onnx",
    }


def validate_quantized_model(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    op_types = {node.op_type for node in model.graph.node}
    quantized_ops = op_types & {"QLinearConv", "QLinearMatMul", "QuantizeLinear"}
    if not quantized_ops:
        raise RuntimeError(f"no quantized operators found in {model_path}")
    if any(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in model.graph.initializer
    ):
        raise RuntimeError(f"external tensor data is not supported: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != list(INPUT_SHAPE):
        raise RuntimeError(
            f"quantized model input is not static {INPUT_SHAPE}: {session.get_inputs()[0].shape}"
        )
    if session.get_outputs()[0].shape != list(OUTPUT_SHAPE):
        raise RuntimeError(
            f"quantized model output is not static {OUTPUT_SHAPE}: {session.get_outputs()[0].shape}"
        )


def quantize_variant(
    model_path,
    output_path,
    calibration_images,
    input_name,
    height,
    width,
    method,
    percentile,
):
    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }
    if method == CalibrationMethod.Percentile:
        extra_options["CalibPercentile"] = percentile

    quantize_static(
        str(model_path),
        str(output_path),
        ImageDataReader(calibration_images, input_name, height, width),
        quant_format=QuantFormat.QOperator,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        extra_options=extra_options,
    )
    validate_quantized_model(output_path)
    print(f"OK: generated {output_path}")


def evaluate_variants(fp32_path, variants, test_images, input_name, height, width):
    sessions = {
        "fp32": ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]),
        **{
            name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for name, path in variants.items()
        },
    }
    statistics = {
        name: {"tensor_mae": [], "output_sum_abs_error": []}
        for name in variants
    }

    for image_path in test_images:
        input_tensor = preprocess(image_path, height, width)
        reference = sessions["fp32"].run(None, {input_name: input_tensor})[0]
        if reference.shape != OUTPUT_SHAPE:
            raise RuntimeError(
                f"model output is {reference.shape}, but this workflow requires {OUTPUT_SHAPE}"
            )
        for name in variants:
            output = sessions[name].run(None, {input_name: input_tensor})[0]
            if output.shape != reference.shape:
                raise RuntimeError(
                    f"output shape mismatch for {name}: {output.shape} != {reference.shape}"
                )
            statistics[name]["tensor_mae"].append(
                float(np.mean(np.abs(output.astype(np.float64) - reference.astype(np.float64))))
            )
            statistics[name]["output_sum_abs_error"].append(
                float(abs(output.astype(np.float64).sum() - reference.astype(np.float64).sum()))
            )

    return {
        name: {
            "test_images": len(test_images),
            "mean_tensor_mae_vs_fp32": float(np.mean(values["tensor_mae"])),
            "mean_output_sum_abs_error_vs_fp32": float(
                np.mean(values["output_sum_abs_error"])
            ),
        }
        for name, values in statistics.items()
    }


def write_results(path, rows):
    fieldnames = [
        "source_model",
        "variant",
        "quantized_model",
        "calibration_images",
        "test_images",
        "mean_tensor_mae_vs_fp32",
        "mean_output_sum_abs_error_vs_fp32",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_quantization_manifest(path, args, model_path, outputs):
    manifest = {
        "source_model": str(model_path),
        "dataset": str(args.dataset.resolve()),
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "calibration_images": args.calibration_count,
        "test_images": args.test_count,
        "random_seed": args.seed,
        "preprocessing": {
            "channel_order": "RGB",
            "layout": "NCHW",
            "pixel_range": [0, 255],
            "mean": DEFAULT_MEAN.reshape(3).tolist(),
            "std": DEFAULT_STD.reshape(3).tolist(),
            "resize": "bilinear",
        },
        "variants": {
            "minmax": {
                "calibration_method": "MinMax",
                "model": outputs["minmax"].name,
            },
            "percentile": {
                "calibration_method": "Percentile",
                "percentile": args.percentile,
                "model": outputs["percentile"].name,
            },
        },
        "quantization": {
            "format": "QOperator",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "activation_symmetric": True,
            "weight_symmetric": True,
            "per_channel": False,
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    images = collect_images(args.dataset.resolve())
    calibration_images, test_images = split_dataset(
        images,
        args.calibration_count,
        args.test_count,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"dataset split: calibration={len(calibration_images)}, "
        f"test={len(test_images)}, seed={args.seed}"
    )

    for model_path in args.models:
        model_path = model_path.resolve()
        input_name, height, width = model_input_spec(model_path)
        model_output_dir = args.output_dir / model_path.stem
        model_output_dir.mkdir(parents=True, exist_ok=True)
        outputs = output_paths(model_path, model_output_dir, args.percentile)

        with tempfile.TemporaryDirectory(prefix="tvm-gemmini-static-onnx-") as temp_dir:
            static_model = Path(temp_dir) / f"{model_path.stem}_static.onnx"
            write_static_model(model_path, static_model)
            quantize_variant(
                static_model,
                outputs["minmax"],
                calibration_images,
                input_name,
                height,
                width,
                CalibrationMethod.MinMax,
                args.percentile,
            )
            quantize_variant(
                static_model,
                outputs["percentile"],
                calibration_images,
                input_name,
                height,
                width,
                CalibrationMethod.Percentile,
                args.percentile,
            )

        evaluation = evaluate_variants(
            model_path,
            outputs,
            test_images,
            input_name,
            height,
            width,
        )
        rows = []
        for name, metrics in evaluation.items():
            print(f"{model_path.name} / {name}: {metrics}")
            rows.append(
                {
                    "source_model": str(model_path),
                    "variant": name,
                    "quantized_model": str(outputs[name]),
                    "calibration_images": len(calibration_images),
                    **metrics,
                }
            )

        results_path = model_output_dir / "evaluation.csv"
        write_results(results_path, rows)
        manifest_path = model_output_dir / "quantization.json"
        write_quantization_manifest(manifest_path, args, model_path, outputs)
        print(f"OK: wrote {results_path}")
        print(f"OK: wrote {manifest_path}")


if __name__ == "__main__":
    main()

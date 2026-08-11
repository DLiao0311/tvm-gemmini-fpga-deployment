import argparse
import os
import tarfile
from datetime import datetime
from pathlib import Path
import onnx
import tvm
from collections import Counter
from tvm import relay
from tvm.relay import ExprMutator
from tvm.relay.dataflow_pattern import DFPatternCallback, is_constant, is_op, wildcard, rewrite
from tvm.contrib import gemmini
from tvm.contrib.gemmini.legalize import LegalizeGemmini

try:
    from tvm.micro import export_model_library_format
except ImportError:
    export_model_library_format = None

SHAPE_DICT = {"input": (1, 3, 512, 512)}


def dump_ir(mod, path):
    # 將目前的 Relay module 輸出成文字，方便逐步檢查各階段結果。
    with open(path, "w", encoding="utf-8") as f:
        f.write(mod.astext(show_meta_data=False))


def rewrite_main(mod, callback):
    # 對 main 套用一次 DFPattern callback，並重新執行 InferType。
    rewritten = rewrite(callback, mod["main"])
    if isinstance(rewritten, relay.Function):
        main = relay.Function(
            params=list(rewritten.params),
            body=rewritten.body,
            ret_type=None,
            type_params=rewritten.type_params,
            attrs=rewritten.attrs,
        )
    else:
        prev = mod["main"]
        main = relay.Function(
            params=list(prev.params),
            body=rewritten,
            ret_type=None,
            type_params=prev.type_params,
            attrs=prev.attrs,
        )
    mod = tvm.IRModule.from_expr(main)
    return relay.transform.InferType()(mod)


class DequantReluQuantToClip(DFPatternCallback):
    # 將 dequantize -> relu -> quantize 改寫成整數域 clip，適用於對稱 int8 流程。

    def __init__(self):
        super().__init__(require_type=True)
        self.q_in = wildcard()
        deq = is_op("qnn.dequantize")(self.q_in, is_constant(), is_constant())
        relu = is_op("nn.relu")(deq)
        self.pattern = is_op("qnn.quantize")(relu, is_constant(), is_constant())

    def callback(self, pre, post, node_map):
        if not isinstance(post, relay.Call):
            return post

        out_dtype = str(post.attrs.out_dtype)
        if out_dtype not in ("int8", "uint8"):
            return post

        zp = post.args[2]
        if not isinstance(zp, relay.Constant):
            return post
        if int(zp.data.numpy().item()) != 0:
            return post

        qmax = 127.0 if out_dtype == "int8" else 255.0
        return relay.clip(node_map[self.q_in][0], a_min=0.0, a_max=qmax)


class NchwToNhwcMutator(ExprMutator):
    # 一次性改寫：將 NCHW 風格的 qnn/nn 呼叫轉成對應的 NHWC 版本。

    def visit_call(self, call):
        new_call = super().visit_call(call)
        if not isinstance(new_call, relay.Call) or not isinstance(new_call.op, tvm.ir.Op):
            return new_call

        op_name = new_call.op.name
        args = list(new_call.args)

        if op_name == "transpose":
            axes = list(getattr(new_call.attrs, "axes", []))
            if axes == [0, 3, 1, 2]:
                return args[0]
            return new_call

        if op_name == "qnn.quantize":
            axis = int(getattr(new_call.attrs, "axis", -1))
            if axis == 1:
                return relay.qnn.op.quantize(
                    args[0], args[1], args[2], axis=3, out_dtype=str(new_call.attrs.out_dtype)
                )
            return new_call

        if op_name == "qnn.dequantize":
            axis = int(getattr(new_call.attrs, "axis", -1))
            if axis == 1:
                return relay.qnn.op.dequantize(args[0], args[1], args[2], axis=3)
            return new_call

        if op_name == "qnn.requantize":
            axis = int(getattr(new_call.attrs, "axis", -1))
            if axis == 1:
                return relay.qnn.op.requantize(
                    args[0], args[1], args[2], args[3], args[4], axis=3, out_dtype=str(new_call.attrs.out_dtype)
                )
            return new_call

        if op_name == "nn.bias_add":
            axis = int(getattr(new_call.attrs, "axis", -1))
            if axis == 1:
                return relay.nn.bias_add(args[0], args[1], axis=3)
            return new_call

        if op_name == "qnn.conv2d":
            data_layout = str(getattr(new_call.attrs, "data_layout", "NCHW"))
            kernel_layout = str(getattr(new_call.attrs, "kernel_layout", "OIHW"))
            if data_layout == "NHWC" and kernel_layout == "HWIO":
                return new_call

            new_kernel = args[1]
            if kernel_layout == "OIHW":
                new_kernel = relay.layout_transform(args[1], src_layout="OIHW", dst_layout="HWIO")

            return relay.qnn.op.conv2d(
                args[0],
                new_kernel,
                args[2],
                args[3],
                args[4],
                args[5],
                strides=tuple(new_call.attrs.strides),
                padding=tuple(new_call.attrs.padding),
                dilation=tuple(new_call.attrs.dilation),
                groups=int(new_call.attrs.groups),
                channels=int(new_call.attrs.channels),
                kernel_size=tuple(new_call.attrs.kernel_size),
                data_layout="NHWC",
                kernel_layout="HWIO",
                out_layout=str(getattr(new_call.attrs, "out_layout", "")),
                out_dtype=str(new_call.attrs.out_dtype),
            )

        return new_call


def normalize_main_signature(mod):
    # 在 mutator 改寫後重建 main，讓參數、body 與 attrs 維持一致。
    main = mod["main"]
    new_main = relay.Function(
        list(main.params),
        main.body,
        ret_type=None,
        type_params=main.type_params,
        attrs=main.attrs,
    )
    return relay.transform.InferType()(tvm.IRModule.from_expr(new_main))


def rewrite_main_with_mutator(mod):
    # 不使用 ConvertLayout，直接套用一次 NCHW -> NHWC 的 mutator。
    main = mod["main"]
    mutator = NchwToNhwcMutator()
    new_body = mutator.visit(main.body)
    new_main = relay.Function(
        list(main.params),
        new_body,
        ret_type=None,
        type_params=main.type_params,
        attrs=main.attrs,
    )
    mod = tvm.IRModule.from_expr(new_main)
    return relay.transform.InferType()(mod)


def switch_main_input_to_nhwc(mod):
    # 將 main 的輸入簽名改成 NHWC，並暫時插入 transpose 作為銜接。
    main = mod["main"]
    old_in = main.params[0]
    n, c, h, w = [int(v) for v in old_in.type_annotation.shape]
    new_in = relay.var(old_in.name_hint, shape=(n, h, w, c), dtype="float32")
    bridged = relay.expr.bind(main.body, {old_in: relay.transpose(new_in, axes=[0, 3, 1, 2])})
    new_main = relay.Function([new_in], bridged, ret_type=None, type_params=main.type_params, attrs=main.attrs)
    mod = tvm.IRModule.from_expr(new_main)
    return relay.transform.InferType()(mod)


def step1_one_shot_layout_rewrite(mod):
    # 流程步驟 1: dataflow NCHW -> NHWC 改寫。
    mod = switch_main_input_to_nhwc(mod)
    mod = rewrite_main_with_mutator(mod)
    #mod = relay.transform.FoldConstant()(mod)
    # mod = normalize_main_signature(mod)
    # dump_ir(mod, "04_after_one_shot_layout.relay")
    return mod


def step2_symmetric_cleanup(mod, dump_path=None):
    # 流程步驟 2：在可成立的情況下，將 dequant -> relu -> quant 清理成整數域 clip。
    mod = rewrite_main(mod, DequantReluQuantToClip())
    mod = relay.transform.SimplifyExpr()(mod)
    # mod = relay.transform.FoldConstant()(mod)
    # mod = relay.transform.InferType()(mod)
    if dump_path is not None:
        dump_ir(mod, dump_path)
    return mod


def gemmini_lower(mod):
    # 合併 Gemmini composite pattern，並 legalize 成 contrib.gemmini op。
    pattern = relay.op.contrib.get_pattern_table("gemmini")
    mod = relay.transform.MergeComposite(pattern)(mod)
    # mod = relay.transform.InferType()(mod)
    # dump_ir(mod, "06_after_gemmini_merge_composite.relay")

    mod = LegalizeGemmini()(mod)
    mod = relay.transform.InferType()(mod)
    # dump_ir(mod, "07_after_gemmini_legalize.relay")
    return mod


def save_mlf_artifacts(mlf_tar, output_dir):
    project_dir = Path(output_dir).resolve()
    mlf_root = project_dir / "runs"
    mlf_dir = mlf_root / datetime.now().strftime("%m%d_%H%M")
    mlf_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(mlf_tar, "r") as tf:
        tf.extractall(mlf_dir)

    print(f"OK: extracted MLF bundle to {mlf_dir}")
    headers = list(mlf_dir.rglob("tvmgen_default.h"))
    if headers:
        print(f"OK: found tvmgen_default.h at {headers[0]}")
    else:
        print("WARN: tvmgen_default.h was not found inside the MLF bundle")
    return mlf_dir


def build_mlf(mod, params, output_dir="."):
    # target: Gemmini Model Library Format?
    os.makedirs(output_dir, exist_ok=True)
    mlf_tar = f"{output_dir}/mlf.tar"

    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    target = tvm.target.Target({"kind": "c", "device": "gemmini"})
    executor = tvm.relay.backend.Executor(
        "aot",
        options={
            "unpacked-api": 1,
            "interface-api": "c",
            "workspace-byte-alignment": 16,
        },
    )
    with gemmini.build_config(usmp_alg="hill_climb", opt_level=3, disabled_pass=["AlterOpLayout"]):
        module = relay.build(mod, target=target, params=params, runtime=runtime, executor=executor)

    if export_model_library_format is None:
        print("WARN: tvm.micro.export_model_library_format is unavailable; skip MLF export")
        return

    export_model_library_format(module, mlf_tar)
    print(f"OK: exported {mlf_tar}")
    save_mlf_artifacts(mlf_tar, output_dir)


def print_op_stats(mod, title):
    # 列出 operator 統計，方便在改寫後快速確認結果是否合理。
    counter = Counter()

    def visit(expr):
        if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
            counter[expr.op.name] += 1

    relay.analysis.post_order_visit(mod["main"], visit)
    print(f"\n===== {title} =====")
    for name, cnt in sorted(counter.items()):
        print(f"{name:25s} : {cnt}")


def load_onnx_model(onnx_path):
    # 載入 ONNX 模型，並轉成帶型別資訊的 Relay IRModule 與 params。
    onnx_model = onnx.load(onnx_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape=SHAPE_DICT, freeze_params=True)
    mod = relay.transform.InferType()(mod)
    return mod, params


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export a quantized ONNX CNN as Gemmini MLF")
    parser.add_argument("--model", required=True, type=Path, help="Input ONNX model")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "generated" / "mlf",
        help="MLF output directory (default: generated/mlf)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mod, params = load_onnx_model(args.model)
    dump_ir(mod, args.output / "00_from_onnx.relay")

    mod = step1_one_shot_layout_rewrite(mod)
    mod = step2_symmetric_cleanup(mod, args.output / "05_after_layout_fuse.relay")
    print_op_stats(mod, "After Step2")

    mod = gemmini_lower(mod)
    build_mlf(mod, params, output_dir=str(args.output))


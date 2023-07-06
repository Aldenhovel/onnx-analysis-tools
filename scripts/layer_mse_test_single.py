import os.path
import numpy as np
from typing import Tuple
import argparse

import onnx
from collections import OrderedDict
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModel import OnnxGraph, load_onnx, save_onnx
from tools.OnnxModel.TestModel import test_model_ort, test_model_r8_mm
from tools.OnnxModelConvert import ModifyGraph
from utils.StyleText import *
from utils.Pickle import load_pkl, save_pkl, check_pkl

def gen_input(shape: Tuple = (3, 224, 224), seed: [Optional] = None) -> np.ndarray:
    if seed:
        np.random.seed(seed)
    x = np.random.randn(*shape)
    return x.astype(np.float32)


def cvt_r8_input(tensor: np.ndarray) -> np.ndarray:
    return tensor.astype(np.float32).ravel()


def mse(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape == y.shape, f"Shape not match: {x.shape} and {y.shape}"
    return float(np.mean((x - y) ** 2))


def write_pkl(ort_out: np.ndarray, r8_out: np.ndarray, out_layer_ix: int, out_layer_name: str, shape: Tuple, pkl_file: str):
    """
    每个层测试完使用一次，递增式记录
    """
    if not os.path.exists(pkl_file):
        data = OrderedDict()
    else:
        data = load_pkl(pkl_file)
    data[out_layer_ix] = {'name': out_layer_name, 'ort': ort_out, 'r8': r8_out, 'diff': np.abs(ort_out - r8_out), 'shape': shape}
    save_pkl(pkl_file, data)


def main(model: onnx.ModelProto, log_file: str):
    TMP_MODEL = "../tmp/subgraph_testing.onnx"
    LOG_PATH = log_file

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            pass

    infer_sequence = OnnxGraph.get_infer_sequence(model)

    layer_skipped = []
    for ix, node in enumerate(infer_sequence):

        try:

            layer_inputs = [input_.name for input_ in infer_sequence[ix].inputs]
            layer_outputs = [output_.name for output_ in infer_sequence[ix].outputs]

            assert len(layer_outputs) == 1, "Now only support Multi-input Single-output."
            print(f"#{ix} {layer_inputs} {layer_outputs}")

            submodel = ModifyGraph.modify_graph_surgeon(model, layer_inputs, layer_outputs)
            save_onnx(submodel, TMP_MODEL)
            input_shapes = OnnxGraph.get_input_shape(submodel)

            del submodel

            # Generate random input and test onnxruntime
            input_tensors, input_tensors_r8 = {}, {}
            for name, shape in input_shapes.items():
                tensor = gen_input(shape)
                input_tensors[name] = tensor

            output_ort = test_model_ort(TMP_MODEL, input_tensors=input_tensors, verbose=False)
            output_ort = output_ort[0]
            output_shape = output_ort.shape
            output_ort = output_ort.ravel()

            output_r8 = test_model_r8_mm(TMP_MODEL, input_tensors)

            print(f"{style_info()} ORT output: {style_info(str(output_ort.shape))}")
            print(f"{style_info()}  R8 output: {style_info(str(output_r8.shape))}")
            assert output_ort.shape == output_r8.shape, f"Shape error between ORT and R8 inference: {output_ort.shape} and {output_r8.shape}"

            # calc mse and make log
            mse_error = mse(output_r8, output_ort)
            print(f"{style_info()} MSE: {mse_error}")

            write_pkl(output_ort, output_r8, ix, layer_outputs[0], output_shape, log_file)
            print("=" * 100)

        except Exception as e:
            print(f"{style_warning()} Skip one layer")
            layer_skipped.append((ix, e))

    if os.path.exists(TMP_MODEL):
        os.remove(TMP_MODEL)
    print(f"{style_pass()} Finish. {len(layer_skipped)} layers skipped.")
    for ix, e in layer_skipped:
        print(f"{style_warning()} {ix}# {e}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="onnx model path", type=str, required=True)
    parser.add_argument('--log', help="log file path. default = '../tmp/diff.pkl'.", type=str,
                        default="../tmp/diff.pkl")

    args = parser.parse_args()

    onnx_path = args.model  # e.g. "../models/resnet18.onnx"
    log_file = args.log

    if os.path.exists(log_file):
        os.remove(log_file)

    model = load_onnx(onnx_path)
    main(model, log_file=log_file)
    print(f"{style_pass()} Finish.")
    print(f"{style_info()} Log saved in {style_info(log_file)}.")

    # show log pkl file
    check_pkl(log_file)



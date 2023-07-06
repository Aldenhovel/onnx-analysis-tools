import os.path
import numpy as np
from typing import Tuple
import argparse

import onnx
import pandas as pd
from collections import OrderedDict
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModel import OnnxGraph, load_onnx, save_onnx
from tools.OnnxModel.TestModel import test_model_ort, test_model_r8
from tools.OnnxModelConvert import ModifyGraph
from utils.StyleText import *
from utils.Pickle import load_pkl, save_pkl, check_pkl

def gen_input(shape: Tuple = (3, 224, 224), seed: [Optional] = None) -> np.ndarray:
    if seed:
        np.random.seed(seed)
    x = np.random.randn(*shape)
    return x


def cvt_r8_input(tensor: np.ndarray) -> np.ndarray:
    return tensor.astype(np.float32).ravel()


def mse(x: np.ndarray, y: np.ndarray) -> float:
    assert x.shape == y.shape, f"Shape not match: {x.shape} and {y.shape}"
    return float(np.mean((x - y) ** 2))


def write_xlsx(ort_out: np.ndarray, r8_out: np.ndarray, out_layer: str, xlsx_file: str):
    """
    每个层测试完使用一次，递增式记录
    """
    mtx = np.stack((ort_out, r8_out, np.abs(ort_out - r8_out)), axis=1)
    df = pd.DataFrame(mtx, columns=['onnxruntime value', 'r8 value', 'diff'])

    if not os.path.exists(xlsx_file):

        with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=out_layer.replace('/', '_'))
    else:
        with pd.ExcelWriter(xlsx_file, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=out_layer.replace('/', '_'))


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



def main(model:onnx.ModelProto, input_tensor_name: str, input_tensor_shape: Tuple, log_file: str):
    TMP_MODEL = "subgraph_testing.onnx"
    LOG_PATH = log_file

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            pass

    infer_sequence = OnnxGraph.get_infer_sequence(model)
    infer_sequence = [node.outputs[0].name for node in infer_sequence]

    for ix, output_tensor_name in enumerate(infer_sequence):
        submodel = ModifyGraph.modify_graph_surgeon(model, [input_tensor_name], [output_tensor_name])
        save_onnx(submodel, TMP_MODEL)

        del submodel

        # Generate random input and test onnxruntime
        input_tensor = gen_input(shape=input_tensor_shape)
        output_ort = test_model_ort(TMP_MODEL,
                                    input_tensors={input_tensor_name: input_tensor[np.newaxis, :].astype(np.float32)},
                                    verbose=False)
        output_ort = output_ort[0]
        output_shape = output_ort.shape
        output_ort = output_ort.ravel()

        # convert the same input data for r8, and test
        input_tensor = cvt_r8_input(input_tensor)
        output_r8 = test_model_r8(TMP_MODEL, input_tensor)

        print(f"{style_info()} ORT output: {style_info(str(output_ort.shape))}")
        print(f"{style_info()}  R8 output: {style_info(str(output_r8.shape))}")
        assert output_ort.shape == output_r8.shape, f"Shape error between ORT and R8 inference: {output_ort.shape} and {output_r8.shape}"

        # calc mse and make log
        mse_error = mse(output_r8, output_ort)
        print(f"{style_info()} MSE: {mse_error}")

        write_pkl(output_ort, output_r8, ix, output_tensor_name, output_shape, log_file)
        print("=" * 100)

    if os.path.exists(TMP_MODEL):
        os.remove(TMP_MODEL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="onnx model path", type=str, required=True)
    parser.add_argument('--input_name', help="onnx model's input name", type=str, required=True)
    parser.add_argument('--input_shape', help="onnx model's input shape", type=int, nargs='+', required=True)
    parser.add_argument('--log', help="log file path. default = '../tmp/diff.pkl'.", type=str, default="../tmp/diff.pkl")

    args = parser.parse_args()

    onnx_path = args.model          # e.g. "../models/resnet18.onnx"
    input_name = args.input_name    # e.g. "input"
    input_shape = args.input_shape  # e.g. (3, 224, 224)
    log_file = args.log

    if os.path.exists(log_file):
        os.remove(log_file)

    model = load_onnx(onnx_path)
    main(model, input_name, input_shape, log_file=log_file)
    print(f"{style_pass()} Finish.")
    print(f"{style_info()} Log saved in {style_info(log_file)}.")

    # show log pkl file
    check_pkl(log_file)





import argparse
import sys
import pathlib
import os
import pandas as pd
sys.path.append(str(pathlib.Path(".").absolute().parent))

from utils.StyleText import *
from tools.OnnxModel import load_onnx, save_onnx
from tools.OnnxModelConvert import ModifyGraph


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_onnx_file', required=True, type=str, help="input onnx file path.")
    parser.add_argument('-o', '--output_onnx_file', type=str, default="", help="output onnx file path.")
    parser.add_argument('-x', '--input_names', type=str, nargs='+', required=True, help="subgraph input(s) name.")
    parser.add_argument('-y', '--output_names', type=str, nargs='+', required=True, help="subgraph output(s) name.")
    parser.add_argument('-v', '--version', type=str, default='api', help="'surgeon' or 'api' version to modify the model.")
    parser.add_argument('-r', '--remove_initializer', type=bool, default=True, help="remove initializer.")

    args = parser.parse_args()
    input_onnx_file = args.input_onnx_file
    if not os.path.exists(input_onnx_file):
        print(f"{style_error()} Onnx file not found: {input_onnx_file}.")
    model = load_onnx(input_onnx_file)

    output_onnx_file = args.output_onnx_file
    if output_onnx_file == "":
        onnx_file_output = input_onnx_file
        print(f"{style_warning()} The modified model will be saved in {onnx_file_output}.")

    xs = args.input_names
    ys = args.output_names

    version = args.version
    rm_init = args.remove_initializer
    if version == 'api':
        if not rm_init:
            print(f"{style_warning()} api version need not to remove initializer.")
        model_ = ModifyGraph.modify_graph_onnxapi(model, xs, ys)
    elif version == 'surgeon':
        model_ = ModifyGraph.modify_graph_surgeon(model, xs, ys, rm_init)
    else:
        print(f"{style_error()} Modify version should be 'api' or 'surgeon', but got {version}.")
        return

    save_onnx(model_, output_onnx_file)
    print(f"{style_pass()} Finish.")


if __name__ == "__main__":
    main()


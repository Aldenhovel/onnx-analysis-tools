import argparse
import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModel import Load, OnnxGraph, Visualize
from tools.OnnxModelConvert import Modify
from utils.StyleText import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-onnx-file", help="input onnx file path.", type=str, required=True)
    parser.add_argument("-o", "--output-onnx-file", help="output onnx file path.", type=str, default="")
    parser.add_argument("-n", "--input-name", help="model's input name.", type=str, required=True)
    parser.add_argument("-s", "--fixed-shape", help="shape of fixed input.", type=int, nargs="+", required=True)

    args = parser.parse_args()
    onnx_path = args.input_onnx_file
    save_path = args.output_onnx_file

    if save_path == "":
        save_path = onnx_path
        print(f"{style_warning()} Model will be saved in {save_path}")

    input_name = args.input_name

    if args.fixed_shape == None:
        print(f"{style_error()} Shape is None.")
        return
    fixed_shape = tuple(args.fixed_shape)

    if not os.path.exists(onnx_path):
        print(f"{style_error()} Onnx file not exist: {onnx_path}")
        return
    model = Load.load_onnx(onnx_path)

    model = Modify.fixed_input(model, save_path=save_path, input_name=input_name, fixed_shape=fixed_shape)
    Visualize.visualize_nodes(model, info=['input_tensor'])

    print(f"{style_pass()} Finish.")



if __name__ == "__main__":
    main()


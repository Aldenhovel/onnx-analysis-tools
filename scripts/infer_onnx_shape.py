import argparse
import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModel import OnnxGraph
from tools.OnnxModel import load_onnx, save_onnx
from utils.StyleText import *


def main(input_onnx_file: str, output_onnx_file: str):
    model = load_onnx(input_onnx_file)
    model_with_shape = OnnxGraph.infer_shape(model)
    save_onnx(model_with_shape, output_onnx_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_onnx_file', type=str, required=True, help="input onnx file path.")
    parser.add_argument('-o', '--output_onnx_file', type=str, default="", help="output onnx file path.")

    args = parser.parse_args()

    input_onnx_file = args.input_onnx_file
    output_onnx_file = args.output_onnx_file

    if len(output_onnx_file) == 0:
        output_onnx_file = input_onnx_file
        print(f"{style_warning()} model will be saved in {output_onnx_file}")
    main(input_onnx_file, output_onnx_file)
    print(f"{style_pass()} Finish.")


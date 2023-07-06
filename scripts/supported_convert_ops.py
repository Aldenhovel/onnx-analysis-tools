import argparse
import sys
import pathlib
import os
import pandas as pd
sys.path.append(str(pathlib.Path(".").absolute().parent))
from tools.OnnxModel import load_onnx, OnnxGraph
from tools.OnnxModelConvert import get_supported_operations
from utils.StyleText import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-onnx-file", help="input onnx file path.", type=str, required=True)

    args = parser.parse_args()
    onnx_path = args.input_onnx_file

    if not os.path.exists(onnx_path):
        print(f"{style_error()} Onnx file not exist: {onnx_path}")
        return
    model = load_onnx(onnx_path)

    config_path = "../tools/OnnxModelConvert/supported_operations.json"
    check, supported_ops = get_supported_operations(config_path)
    if not check:
        return

    nodes = OnnxGraph.get_node_set(model)
    node_dict = OnnxGraph.get_node_dict(model)

    supported, unsupported = [], []
    for node in nodes:
        if node in supported_ops:
            supported.append(f"{node} ({style_pass(str(node_dict[node]))})")
        else:
            unsupported.append(f"{node} ({style_warning(str(node_dict[node]))})")


    if len(unsupported) == 0:
        print(f"{style_info()} All operation or function is supported in this model.")
    elif len(supported) == 0:
        print(f"{style_info()} No operation or function is supported in this model.")
    else:
        pass
    df_supported = pd.DataFrame({f"Supported": supported})
    df_unsupported = pd.DataFrame({"Unsupported": unsupported})
    print("=" * 50)
    print("Supported Operations:")
    print(df_supported.to_string(header=False) if len(supported) > 0 else "None")
    print("-" * 50)
    print("Unsupported Operations:")
    print(df_unsupported.to_string(header=False) if len(unsupported) > 0 else "None")

    print("=" * 50)
    print(f"{style_pass()} Finish.")


if __name__ == "__main__":
    main()

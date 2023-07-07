import argparse
import sys
import pathlib
import os
import pandas as pd
sys.path.append(str(pathlib.Path(".").absolute().parent))
from tools.OnnxModelConvert import Simplify
from tools.OnnxModel import load_onnx, OnnxGraph
from utils.StyleText import *


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-onnx-file", help="input onnx file path.", type=str, required=True)
    parser.add_argument("-o", "--output-onnx-file", help="output onnx file path.", type=str, default="")

    args = parser.parse_args()
    onnx_path = args.input_onnx_file
    save_path = args.output_onnx_file

    if not os.path.exists(onnx_path):
        print(f"{style_error()} Onnx file not exist: {onnx_path}")
        return
    model = load_onnx(onnx_path)

    if save_path == "":
        save_path = onnx_path
        print(f"{style_warning()} Model will be saved in {save_path}")

    original_node_dict = OnnxGraph.get_node_dict(model)
    original_size = round(OnnxGraph.get_model_size(model) / 1024**2, 4)
    df_org = pd.DataFrame.from_dict(original_node_dict, orient='index', columns=['Original'])
    df_org.reset_index(inplace=True)
    df_org.rename(columns={'index': 'Node'}, inplace=True)

    print(f"{style_info()} Simplifying onnx model ... ")
    model = Simplify.simplify_onnx(model=model, save_path=save_path)

    simplified_node_dict = OnnxGraph.get_node_dict(model)
    simplified_size = round(OnnxGraph.get_model_size(model) / 1024**2, 4)
    df_sim = pd.DataFrame.from_dict(simplified_node_dict, orient='index', columns=['Simplified'])
    df_sim.reset_index(inplace=True)
    df_sim.rename(columns={'index': 'Node'}, inplace=True)
    df = df_org.merge(df_sim, on='Node', how='outer')
    df.fillna(0, inplace=True)
    df = df.sort_values(by=['Node'])
    df['Original'] = df['Original'].astype(int)
    df['Simplified'] = df['Simplified'].astype(int)
    df[''] = df['Simplified'] - df['Original']

    for index, row in df.iterrows():
        if row[''] != 0:
            df.at[index, ''] = style_pass(str(row['']))
        else:
            df.at[index, ''] = ''

    print(f"{style_pass()} Done, nodes difference:\n")
    print("=" * 50)

    # df.reset_index(inplace=True)
    print(df.to_string())
    print("-" * 50)

    print(f"Original Model Size: \t{original_size} MiB")
    print(f"Simplified Model Size: \t{simplified_size} MiB ({style_pass(str(round(simplified_size - original_size, 4)))} MiB)")
    print("=" * 50)


if __name__ == "__main__":
    main()

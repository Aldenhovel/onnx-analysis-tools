import sys
import pathlib
import onnx
from typing import List, Dict, Optional
import pandas as pd
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModel import OnnxGraph
from tools.OnnxModel import load_onnx
from utils.StyleText import *


def visualize_nodes(model: onnx.ModelProto, info: List[str]=['count', 'input_tensor', 'class', 'node_list']) -> Optional:

    node_list = OnnxGraph.get_node_list(model)
    node_dict = OnnxGraph.get_node_dict(model)

    if 'count' in info:
        print("=" * 50)
        print(f"Nodes: {sum(node_dict.values())}")

    if 'input_tensor' in info:
        print("=" * 50)
        input_tensors = []
        for input_tensor in model.graph.input:
            input_tensor_shape = input_tensor.type.tensor_type.shape
            input_tensor_shape = [dim.dim_value for dim in input_tensor_shape.dim]
            input_tensors.append({'Tensor Name': input_tensor.name,
                                  'Tensor Shape': input_tensor_shape,
                                  '': style_warning("Dynamic shape") if 0 in input_tensor_shape else ''})
        df = pd.DataFrame(input_tensors)
        print(df.to_string())

    if 'class' in info:
        print("=" * 50)
        df = pd.DataFrame.from_dict(node_dict, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Type'}, inplace=True)
        print(df.to_string())

    if 'node_list' in info:
        print("=" * 50)
        df = pd.DataFrame(node_list)
        df = df[['type', 'name']]
        print(df.to_string())

    print("=" * 50)


if __name__ == "__main__":
    model = load_onnx("../../models/subgraph_api.onnx")
    print(model.graph.input)
    visualize_nodes(model)

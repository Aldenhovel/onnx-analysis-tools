import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent.parent))
import onnx
import onnx_graphsurgeon as gs

from tools.OnnxModel.Load import load_onnx
from typing import List, Set, Dict, Any, Optional
from onnxsimUtils import model_info
from utils.StyleText import *

class OnnxGraph:
    def __init__(self):
        pass

    @staticmethod
    def get_node_list(model: onnx.ModelProto) -> List[Dict[str, str]]:
        """
        return onnx model nodes' **name and type** as a list.
        :param model: onnx model.
        :return: all the nodes' name and type in list.
        """
        nodes = []
        for node in model.graph.node:
            nodes.append({'name': node.name, 'type': node.op_type})
        return nodes

    @staticmethod
    def get_node_set(model: onnx.ModelProto) -> Set[str]:
        """
        return onnx model nodes' **type** as a set.
        :param model: onnx model.
        :return: all the nodes' name and type in list.
        """
        nodes = set()
        for node in model.graph.node:
            nodes.add(node.op_type)
        return nodes

    @staticmethod
    def get_node_dict(model: onnx.ModelProto) -> Dict:
        """
        return onnx model's node type counting summary
        """
        node_dict = {}
        info = model_info.ModelInfo(model)
        for key in sorted(list(set(info.op_nums.keys()))):
            node_dict[key] = info.op_nums[key]
        return node_dict

    @staticmethod
    def get_model_size(model: onnx.ModelProto) -> Any:
        info = model_info.ModelInfo(model)
        return info.model_size

    @staticmethod
    def get_input_shape(model: onnx.ModelProto) -> Dict[Any, List]:
        input_tensors = {}
        for input_tensor in model.graph.input:
            print(input_tensor)
            input_tensor_shape = input_tensor.type.tensor_type.shape
            input_tensor_shape = [dim.dim_value for dim in input_tensor_shape.dim]
            input_tensors[input_tensor.name] = input_tensor_shape
        return input_tensors

    @staticmethod
    def is_fixed_input(model: onnx.ModelProto) -> bool:
        for input_tensor in model.graph.input:
            input_shape = input_tensor.type.tensor_type.shape
            input_shape = [dim.dim_value for dim in input_shape.dim]
            if 0 in input_shape:
                return False
        return True

    @staticmethod
    def get_infer_sequence(model: onnx.ModelProto) -> List[str]:

        graph = gs.import_onnx(model)
        graph.toposort()
        sequence = []
        with graph.node_ids():
            for i in range(len(graph.nodes)):
                node = graph.nodes[i]
                sequence.append(node)
        return sequence

    @staticmethod
    def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
        if model.ir_version < 4:
            print(f"{style_info()} Model with ir_version below 4 requires to include initilizer in graph input")
            return model

        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
        return model

    @staticmethod
    def infer_shape(model: onnx.ModelProto) -> onnx.ModelProto:
        from onnx.shape_inference import infer_shapes
        return infer_shapes(model)

if __name__ == "__main__":
    model_path = '../../models/resnet18.onnx'
    model = load_onnx(model_path)

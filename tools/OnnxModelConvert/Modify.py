import onnxruntime.tools.make_dynamic_shape_fixed as fx
import onnx
import onnxruntime
import os
from typing import Tuple, Optional

from tools.OnnxModel import save_onnx, load_onnx, OnnxGraph, Visualize
from utils.StyleText import *


def fixed_input(model: onnx.ModelProto, save_path: str, input_name: str, fixed_shape: Tuple[int]) -> onnx.ModelProto:
    """
    Making onnx model's dynamic input tensor shape fixed.
    """

    if OnnxGraph.is_fixed_input(model):
        print(f"{style_warning()} This model already had fixed input(s) shape.")
        return model
    fx.make_input_shape_fixed(model.graph, input_name=input_name, fixed_shape=fixed_shape)
    save_onnx(model, save_path)
    return model


if __name__ == "__main__":

    model = load_onnx("../../models/resnet18.onnx")
    save_path = "../../models/resnet18.onnx"
    fixed_input(model, save_path=save_path, input_name="input", fixed_shape=(5, 3, 224, 224))

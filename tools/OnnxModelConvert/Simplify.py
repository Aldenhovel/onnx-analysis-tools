import os.path
import json
import onnx
from typing import List, Tuple

from tools.OnnxModel import load_onnx, save_onnx
from utils.StyleText import *
import onnxsim


def simplify_onnx(model: onnx.ModelProto, save_path: str) -> onnx.ModelProto:

    model_opt, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    save_onnx(model_opt, save_path)
    return model_opt


def get_supported_operations(config: str) -> Tuple[bool, List]:

    if not os.path.exists(config):
        print(f"{style_error()} Config file not found.")
        return False, []
    with open(config, 'r') as f:
        data = json.load(f)
        support_ops = data['ops']
    return True, support_ops


if __name__ == "__main__":
    model = load_onnx("../../models/resnet18.onnx")
    simplify_onnx(model, "../../models/resnet18.onnx")

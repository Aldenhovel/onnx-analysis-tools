import onnx
from onnx.external_data_helper import convert_model_to_external_data
import re

from tools.OnnxModel.OnnxGraph import OnnxGraph
from utils.StyleText import *


def save_onnx(model: onnx.ModelProto, onnx_path: str) -> Optional:

    model_size = round(OnnxGraph.get_model_size(model) / 1024 ** 3, 4)  # GiB

    if model_size >= 2:
        external_data_path = re.sub(r"\.onnx", ".data", onnx_path)
        print(
            f"{style_warning()} The model size is {model_size} GiB (>2 GiB), "
            f"external data will be saved in '{external_data_path}'")
        convert_model_to_external_data(model, location=external_data_path)

    onnx.save(model, onnx_path)

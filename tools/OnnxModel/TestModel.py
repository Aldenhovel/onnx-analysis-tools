import onnx.shape_inference
import onnxruntime
import numpy as np
from typing import Optional, Dict, List
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent.parent))

from utils.StyleText import *
from typing import Tuple



def test_model_ort(model_path: str, input_tensors: Dict={}, verbose: bool=True) -> List:

    session = onnxruntime.InferenceSession(model_path)
    if len(input_tensors.items()) == 0:
        if verbose:
            print(f"{style_info()} Generating inputs ...")
        input_tensors = {}
        for tensor in session.get_inputs():
            input_name = tensor.name
            input_shape = tensor.shape
            input_data = np.random.random(input_shape).astype(np.float32)
            input_tensors[input_name] = input_data

    if verbose:
        print("=" * 50)
        print("Model Input:")
        for input_name, input_data in input_tensors.items():
            print(f"input {input_name}: {input_data.shape}")

        print("-" * 50)
    outputs = session.run(None, input_tensors)

    if verbose:
        print("Model Output:")
        for ix, tensor in enumerate(outputs):
            print(f"output[{ix}]: {tensor.shape}")

        print("=" * 50)
    print(f"{style_pass()} Pass onnxruntime testing.")
    return outputs


def gen_input(shape: Tuple = (3, 224, 224)):
    """
    产生一个 shape = (C, H, W) 的随机 np.ndarray 作为输入
    """
    dummy_x = np.random.randn(*shape)
    return dummy_x



if __name__ == '__main__':
 pass



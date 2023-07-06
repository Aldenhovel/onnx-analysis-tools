import argparse
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent))

from tools.OnnxModelConvert.FormatConvert import cvt_cfg2onnx
from tools.OnnxModel.TestModel import test_model_ort
from tools.OnnxModel import save_onnx, OnnxGraph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', help="cfg file path.", type=str, required=True)
    parser.add_argument('-w', '--weights', help="weights file path.", type=str, required=True)
    parser.add_argument('-o', '--output', help="output onnx file path.", type=str, required=True, )
    parser.add_argument('-n', '--neck', help="neck, PAN or FPN", type=str, default='PAN')
    parser.add_argument('-s', '--strides', help="YOLO model cell size", nargs='+', type=int, default=[])

    args = parser.parse_args()
    cfg_file = args.cfg
    weights = args.weights
    output = args.output
    neck = args.neck
    strides = args.strides

    yolo_model = cvt_cfg2onnx(cfg_file, weights, output, neck, strides)
    yolo_model = OnnxGraph.remove_initializer_from_input(yolo_model)

    save_onnx(yolo_model, output)
    test_model_ort(output)

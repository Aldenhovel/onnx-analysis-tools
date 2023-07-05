import onnx
import onnxruntime


def load_onnx(onnx_path: str) -> onnx.ModelProto:
    model = onnx.load(onnx_path)
    return model


def load_session(onnx_path: str) -> onnxruntime.InferenceSession:
    session = onnxruntime.InferenceSession(onnx_path)
    return session


if __name__ == "__main__":
    sess = load_session("../../models/resnet18.onnx")
    assert sess

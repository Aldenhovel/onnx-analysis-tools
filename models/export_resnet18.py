from torchvision.models import resnet18
import torch

if __name__ == "__main__":
    net = resnet18()
    model_name = "resnet18.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {'input':  {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                    }
    #torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)
    torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
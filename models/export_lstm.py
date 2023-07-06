import torch
import torch.nn as nn

if __name__ == "__main__":
    lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2)
    model_name = "lstm.onnx"
    dummy_x, dummy_h, dummy_c = torch.randn(2, 512), torch.randn(2, 512), torch.randn(2, 512)

    input_shape = (dummy_x.shape, dummy_c.shape, dummy_h.shape)

    torch.onnx.export(lstm,
                      (dummy_x, (dummy_c, dummy_h)),
                      model_name,
                      input_names=["input_x", "c", "h"],
                      output_names=["output"],
                      dynamic_axes={'input_x': {0: 'batch_size'}, 'c': {0: 'batch_size'}, 'h': {0: 'batch_size'}}
                      )
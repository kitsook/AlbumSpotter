import ai_edge_torch
import numpy
import torch

from config import config

MODEL_TIMESTAMP = '20241129042637'

# load the torch model
torch_model_file = config['OUTPUT_MODEL_FOLDER'] + "model_" + MODEL_TIMESTAMP + ".pt"
torch_model = torch.load(torch_model_file, weights_only=False)
torch_model.eval()

# random inputs for tracing and shape inference
sample_inputs = (torch.randn(1, 3, 224, 224),)

edge_model = ai_edge_torch.convert(torch_model.eval(), sample_inputs)
edge_model.export(config['OUTPUT_MODEL_FOLDER'] + "model_" + MODEL_TIMESTAMP + ".tflite")

# compare output from the two models
torch_output = torch_model(*sample_inputs)
edge_output = edge_model(*sample_inputs)

if (numpy.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something wrong with Pytorch --> TfLite")

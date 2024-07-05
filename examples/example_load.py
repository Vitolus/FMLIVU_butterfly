import torch
import onnxruntime
from torch import nn
import torch.onnx as onnx
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    model = NeuralNetwork()
    model.load_state_dict(torch.load("../data/model.pth"))
    model.eval()
    input_image = torch.zeros((1, 28, 28))
    onnx_model = 'data/model.onnx'
    onnx.export(model, input_image, onnx_model)

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    x, y = test_data[0][0], test_data[0][1]

    session = onnxruntime.InferenceSession(onnx_model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: x.numpy()})
    predicted, actual = classes[result[0][0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

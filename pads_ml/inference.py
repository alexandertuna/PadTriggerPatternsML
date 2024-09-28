import torch

from pads_ml.model import OneHotFullyConnected

class Inference:
    def __init__(self, model_path: str) -> None:
        self.model = OneHotFullyConnected()
        self.model_path = model_path
        self.model.load_state_dict(model_path)
        self.model.eval()

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

#!/usr/bin/env python

import platform
import numpy as np
import torch as tt
import torch.nn as nn
import torch.nn.functional as F

from rex_xai.input.input_data import Data
from rex_xai.prediction import from_pytorch_tensor

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=3))
        self.layer2 = nn.Sequential(
            nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(40, 40, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1))
        self.fc1 = nn.Linear(8640, 60)
        self.fc2 = nn.Linear(60, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.2)
        out = self.fc2(out)
        return out

model = ConvNet().to('mps')

model.load_state_dict(tt.load("simple_DNA_model.pt", map_location='mps', weights_only=True))
model.eval()


if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")

def preprocess(path, shape, device, mode) -> Data:
    path = tt.from_numpy(np.load(path)).float().unsqueeze(0).unsqueeze(0).to('mps')
    data = Data(path , (1, 1356), 'mps', mode="spectral", process=True)
    data.set_width(1356)
    data.set_height(1)
    return data

def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor, target=target)

def model_shape():
    return ["N", 1, 1356]


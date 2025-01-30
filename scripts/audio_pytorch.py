#¡/usr/bin/env python

import numpy as np
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor
import platform
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
import torch as tt
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=58, out_features=256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(in_features=128, out_features=10),
    nn.ReLU()
).to('mps')


model.load_state_dict(tt.load("/Users/k2261934/Documents/datasets/MusicGenre/audio.pt", weights_only=True))
model.eval()

# df = pd.read_csv("~/Documents/datasets/MusicGenre/file.csv")
# minmax = MinMaxScaler()
#
# X = df.drop(['label', 'filename'], axis=1)
# np_scaled = minmax.fit_transform(X)
#
# audio = np_scaled[0]
#
# np.save("blues.npy", audio)



if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")


def preprocess(path, shape, device, mode) -> Data:
    path = np.load(path)
    data = Data(path , (1, 58), 'mps', mode="spectral", process=True)
    data.set_width(58)
    data.set_height(1)
    return data

def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants.unsqueeze(0))
        if raw:
            return F.softmax(tensor, dim=1)
        z = from_pytorch_tensor(tensor, target=target)
        return z

def model_shape():
    return ["N", 58]


# this script is taken from https://github.com/TheophileBERTELOOT/ARM-AE/blob/master/AutoEncoder.py.
# Berteloot, Théophile, Richard Khoury, and Audrey Durand. "Association rules mining with auto-encoders." International
# Conference on Intelligent Data Engineering and Automated Learning. Cham: Springer Nature Switzerland, 2024.

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, dataSize):
        super(AutoEncoder, self).__init__()
        self.dataSize = dataSize
        outputLayer = nn.Tanh()
        self.encoder = nn.Sequential(
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
            nn.Linear(self.dataSize, self.dataSize),
            outputLayer,
        )

    def save(self, p):
        torch.save(self.encoder.state_dict(), p + 'encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'decoder.pt')

    def load(self, encoderPath, decoderPath):
        self.encoder.load_state_dict(torch.load(encoderPath))
        self.decoder.load_state_dict(torch.load(decoderPath))
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = torch.tensor(x[None, :])
        x = self.encoder(x)
        x = self.decoder(x)
        return x

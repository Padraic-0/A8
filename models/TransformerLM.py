import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        j = torch.arange(0, self.d_model, 2).float().to(x.device)
        f = torch.exp(-(math.log(10000)) * j/self.d_model)
        t = torch.arange(0, x.shape[1], 1).float().unsqueeze(1).to(x.device)
        pe = torch.zeros(x.shape[1], self.d_model).to(x.device)
        pe[:, 0::2] = torch.sin(t * f)
        pe[:, 1::2] = torch.cos(t * f)
        pe = pe.unsqueeze(0)
        return x + pe


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=True)
        self.classifier = nn.Linear(d_model, vocab_size)

    def generateCausalMask(self, L):
        mask = torch.zeros(L,L)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        x = self.embeddings(x)
        x = self.position(x)
        mask = self.generateCausalMask(x.shape[1]).to(x.device)
        x = self.encoder(x, mask, is_causal=True)
        x = self.classifier(x)
        return x

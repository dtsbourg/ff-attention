#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ff-attention in Pytorch package, created by
# Dylan Bourgeois (@dtsbourg)
#
# License: MIT (see LICENSE.md)
#
# Original work by Colin Raffel and Daniel P. Ellis
# "Feed-Forward Networks with Attention Can
# Solve Some Long-Term Memory Problems"
# https://arxiv.org/abs/1512.08756
# (Licensed under CC-BY)

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

class FFAttention(torch.nn.Module):
    def __init__(self, batch_size=10, T=10, D_in=2, D_out=1, hidden=100):
        super(Attention, self).__init__()
        # Net Config
        self.T = T
        self.batch_size = batch_size
        self.n_features = D_in
        self.out_dim = D_out
        self.hidden = hidden

    # Step 1:
    # Compute embeddings h_t for element of sequence x_t
    def embedding(self, x_t):
        raise NotImplementedError

    # Step 2:
    # Compute the embedding activations e_t
    def activation(self, h_t):
        raise NotImplementedError

    # Step 3:
    # Compute the probabilities alpha_t
    def attention(self, e_t):
        softmax = torch.nn.Softmax(dim=0)
        alphas = softmax(e_t).view(self.batch_size, self.out_dim, self.T).repeat(1, self.n_features, 1)
        return alphas

    # Step 4:
    # Compute the context vector c
    def context(self, alpha_t, x_t):
        return torch.bmm(alpha_t, x_t)

    # Step 5:
    # Feed-forward prediction based on c
    def out(self, c):
        raise NotImplementedError

    def forward(self, x, training=True):
        x = self.embedding(x)
        x = self.activation(x)
        alpha = self.attention(x)
        x = self.context(alpha, x)
        x = self.out(x)
        return x, alpha

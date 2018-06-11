#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ff-attention in Pytorch package, created by
# Dylan Bourgeois (@dtsbourg)
#
# License: MIT (see LICENSE.md)
#
# Logger:
# Contains some information about the state

class AttentionState():
    def __init__(self, inputs, alphas, prediction, label):
        self.inputs = inputs
        self.alphas = alphas
        self.prediction = prediction
        self.label = label

class AttentionLog():
    def __init__(self):
        self.losses = {'train': [], 'test': []}
        self.best_epoch = 0
        self.attention_state = None

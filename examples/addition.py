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
# https://arxiv.org/abs/1512.08756 [1]
# (Licensed under CC-BY)
#
#
# The addition Problem is a sample problem for testing LSTMs
# They are defined in the package [lstm_problems](https://github.com/craffel/lstm_problems)
#
# Problem: Let x be the following 2-D vector
# [0.11 0.25 -0.32 ... 0.54 0.21]
# [  0    1     0  ...   1    0 ]
# The goal is to predict the sum of the two values for which the second
# dimension is 1:
# x1 = 0.25 ; x2 = 0.54 in this case.
import sys
sys.path.append('./lstm_problems/lstm_problems')
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from ff_attention import FFAttention
from lstm_problems import lstm_problems
from utils import *


class AdditionAttention(FFAttention):
    def __init__(self, *args, **kwargs):
        super(AdditionAttention, self).__init__(*args, **kwargs)
        self.layer0 = torch.nn.Linear(self.n_features, self.hidden)
        self.layer1 = torch.nn.Linear(self.hidden, self.out_dim)
        self.layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def embedding(self, x_t):
        x_t = self.layer0(x_t)
        return F.leaky_relu(x_t)

    def activation(self, h_t):
        return F.tanh(self.layer1(h_t))

    def out(self, c):
        x = F.leaky_relu(self.layer2(c))
        return F.leaky_relu(self.out_layer(x))

class AdditionSequenceDataset(Dataset):
    def __init__(self, T, n_seqs):
        problem = self.sequence_generator(T, n_seqs)

        self.data = torch.from_numpy(problem[0])
        self.target = torch.from_numpy(problem[1])

    def sequence_generator(self, T, n_seqs):
        return lstm_problems.add(min_length=T, max_length=T, n_sequences=n_seqs)

    def sequence_generator_old(self, T):
        rand_idx = np.random.randint(0, size=2,high=T)
        rand_val = np.random.uniform(low=-1, size=T)
        target_mask = np.zeros(T, dtype=bool)
        target_mask[rand_idx] = True
        problem = np.concatenate(([rand_val], [target_mask]), axis=0)
        yield problem, rand_val, target_mask

    def __getitem__(self, index):
        batch_labels = self.target[index]
        batch_data = self.data[index].float()
        return (batch_data, batch_labels)

    def __len__(self):
        return len(self.data)


def main():
    """
    Run experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = AttentionLog()

    batch_size = 100    # Number of samples in each batch
    lr = 0.003          # Learning rate
    n_seqs = 1000        # number of sequences to generate
    T = 100             # Sequence length
    epoch_num = 50      # Number of epochs to train the network
    show_results = True # Set to false to hide plots

    # Create the model
    model = AdditionAttention(batch_size=batch_size, T=T)

    # calculate the number of batches per epoch
    batch_per_ep = n_seqs // batch_size

    # Using the details from the paper [1]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999))

    # Define training data
    seq_ds = AdditionSequenceDataset(T=T, n_seqs=n_seqs)
    # Define training data loader
    seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Define test data
    seq_ds_test = AdditionSequenceDataset(T=T, n_seqs=n_seqs)
    # Define test data loader
    test_seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds_test,
                                                          batch_size=batch_size,
                                                          shuffle=True)

    preds = []; gt = []; attentions = [];
    for ep in range(epoch_num):  # epochs loop
        batch_losses = [];
        for batch_idx, data in enumerate(seq_dataset_loader): # batches loop
            features, labels = data

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output, alphas = model(features, training=True)
            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

            # Backward pass and updates
            loss.backward()                 # calculate the gradients (backpropagation)
            optimizer.step()                # update the weights

        if not ep % 10:
            print('[TRAIN] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        logger.losses['train'].append(np.mean(batch_losses))

        batch_losses = []; outputs = []; attention = []; label_log = [];
        for batch_idx, data in enumerate(test_seq_dataset_loader):
            features, labels = data
            output, alphas = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            attention.append(alphas.tolist())
            label_log.append(labels.tolist())

            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

        if not ep % 10:
            print('[TEST] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        logger.losses['test'].append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(logger.losses['test']):
            logger.best_epoch = ep
            logger.attention_state = AttentionState(alphas=alphas, inputs=features, label=flatten(label_log), prediction=flatten(outputs))

    print("=== Best epoch:")
    print("Epoch #", logger.best_epoch)
    print("Test Loss = ", logger.losses['test'][logger.best_epoch])
    print("Train Loss = ", logger.losses['train'][logger.best_epoch])

    if show_results is True:
        preds_fl = flatten(logger.attention_state.prediction)
        gt_fl = logger.attention_state.label

        ############################################################
        # LOSS
        plt.figure(figsize=(15,10))
        plt.subplot(3,1,1)
        plot_loss(logger)
        # Sample Predictions
        plt.subplot(3,1,2)
        plot_predictions(y_true=gt_fl, y_pred=preds_fl, sample=50)
        # Error distribution
        plt.subplot(3,1,3)
        plot_error(y_true=gt_fl, y_pred=preds_fl)


        ############################################################
        # ATTENTION
        plot_attention(logger, y_true=gt_fl, y_pred=preds_fl, n_seq=4)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

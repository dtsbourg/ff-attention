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
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import numpy as np

import sys
sys.path.append('..')
from ff_attention import FFAttention
import lstm_problems

flatten = lambda l: [item for sublist in l for item in sublist]

class AdditionAttention(FFAttention):
    def __init__(self, *args, **kwargs):
        super(AdditionAttention, self).__init__(*args, **kwargs)
        self.layer1 = torch.nn.Linear(self.n_features, self.out_dim)
        self.layer3 = torch.nn.Linear(self.n_features, self.hidden)
        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def embedding(self, x_t):
        # Initial = identity
        # h_t = f(x_t), f = id
        return x_t

    def activation(self, h_t):
        batch_norm = torch.nn.BatchNorm1d(self.n_features)
        return F.selu(self.layer1(h_t))

    def out(self, c):
        c_= c.view(self.batch_size, self.out_dim, self.n_features)
        x = F.selu(self.layer3(c_))
        return F.tanh(self.out_layer(x))

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
    device = torch.device('cpu')

    batch_size = 1000   # Number of samples in each batch
    lr = 0.01        # Learning rate
    n_seqs = 1000
    T = 100

    model = AdditionAttention(batch_size=batch_size, T=T)

    # calculate the number of batches per epoch
    batch_per_ep = n_seqs // batch_size

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999))

    # Define custom dataset
    seq_ds = AdditionSequenceDataset(T=T, n_seqs=n_seqs)
    # Define data loader
    seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Define custom dataset
    seq_ds_test = AdditionSequenceDataset(T=T, n_seqs=n_seqs)
    # Define data loader
    test_seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds_test,
                                                          batch_size=batch_size,
                                                          shuffle=True)

    epoch_num = 1000    # Number of epochs to train the network
    losses_train = [];
    losses_test = []; preds = []; gt = []; attentions = [];
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

        if not ep % 100:
            print('[TRAIN] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        losses_train.append(np.mean(batch_losses))

        batch_losses = []; outputs = []; attention = []; label_log = [];
        for batch_idx, data in enumerate(test_seq_dataset_loader):
            features, labels = data
            output, alphas = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            attention.append(alphas.tolist())
            label_log.append(labels.tolist())

            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

        if not ep % 100:
            print('[TEST] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        losses_test.append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(losses_test):
            preds = flatten(outputs)
            gt = flatten(label_log)
            attentions = flatten(attention)

if __name__ == '__main__':
    main()

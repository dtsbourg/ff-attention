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

import sys
sys.path.append('../..')

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from ff_attention import FFAttention
from lstm_problems import lstm_problems
from logger import AttentionLog, AttentionState

flatten = lambda l: np.asarray([item for sublist in l for item in sublist])
module_seperator = lambda x: np.asarray([x[i:i+4] for i in range(0,len(x),4)])

class VPAttention(FFAttention):
    def __init__(self, *args, **kwargs):
        super(VPAttention, self).__init__(*args, **kwargs)
        self.layer0a = torch.nn.Linear(self.n_features, self.hidden)
        self.layer0b = torch.nn.Linear(self.hidden, self.n_features)
        self.layer1 = torch.nn.Linear(self.n_features, self.out_dim)
        self.layer3 = torch.nn.Linear(self.n_features, self.hidden)
        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def embedding(self, x_t):
        #x_t = self.layer0a(x_t)
        #x_t = F.leaky_relu(self.layer0b(x_t))
        return x_t

    def activation(self, h_t):
        #batch_norm = torch.nn.BatchNorm1d(self.n_features)
        return F.leaky_relu(self.layer1(h_t))

    def out(self, c):
        c_= c.view(self.batch_size, self.out_dim, self.n_features)
        x = F.leaky_relu(self.layer3(c_))
        return self.out_layer(x)

class VPSequenceDataset(Dataset):
    def __init__(self, vp_module_readouts, npvs):
        self.data = torch.from_numpy(vp_module_readouts)
        self.target = torch.from_numpy(npvs)

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

    cache = True
    if cache == False:
        # Load data
        npvs_file = 'data/minbias10.npy'
        # Contains the occupancies for every VP sensor
        VP_file  = 'data/VP_occupancies_minbias10.npy'

        dfpv = pd.DataFrame(np.load(npvs_file))

        vp_occ = list(map(np.array, np.load(VP_file)))
        dfvp = pd.DataFrame(dtype=float)
        for i in range(len(vp_occ[:-1])):
            dfvp[i] = list(map(float,vp_occ[i]))
        dfvp = dfvp.transpose()
        # Scale data
        scaler_vp = StandardScaler()
        dfvp = scaler_vp.fit_transform(dfvp)

        joblib.dump(scaler_vp, 'scaler_vp.dump')

        # Create data
        scaler_pv = StandardScaler()
        npvs = scaler_pv.fit_transform(np.asarray(dfpv['EventpRecVertexPrimary']).reshape(-1,1))
        joblib.dump(scaler_pv, 'scaler_pv.dump')
        vp_module_readouts = np.asarray([module_seperator(_) for _ in dfvp])

        # Cache data
        joblib.dump(npvs, 'npvs.dump')
        joblib.dump(vp_module_readouts, 'vp_module_readouts.dump')
    else:
        npvs = joblib.load('npvs.dump')
        vp_module_readouts = joblib.load('vp_module_readouts.dump')
        scaler_pv = joblib.load('scaler_pv.dump')

    batch_size = 100    # Number of samples in each batch
    lr = 0.01           # Learning rate
    n_seqs = 9000       # number of sequences == number of samples
    T = 52              # Sequence length == number of modules in our case
    D_in = 4            # 4 modules per sensor

    npvs = npvs[:9000]
    vp_module_readouts = vp_module_readouts[:9000,:]

    # Create the model
    model = VPAttention(batch_size=batch_size, T=T, D_in=D_in, D_out=1, hidden=208)

    # calculate the number of batches per epoch
    batch_per_ep = n_seqs // batch_size

    # Using the details from the paper [1]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Define training data
    seq_ds = VPSequenceDataset(vp_module_readouts, npvs)
    # Define training data loader
    seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    # Define test data
    seq_ds_test = VPSequenceDataset(vp_module_readouts, npvs)
    # Define test data loader
    test_seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds_test,
                                                          batch_size=batch_size,
                                                          shuffle=False)

    epoch_num = 1    # Number of epochs to train the network
    preds = []; gt = [];
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

        if not ep % 5:
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

        if not ep % 5:
            print('[TEST] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        logger.losses['test'].append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(logger.losses['test']):
            logger.best_epoch = ep
            logger.attention_state = AttentionState(alphas=alphas, inputs=features, label=flatten(label_log), prediction=flatten(outputs))

    # Set to false to hide plots
    show_results = True
    if show_results is True:
        plt.style.use('ggplot')
        plt.figure(1,figsize=(15,10))
        plt.subplot(3,1,1)
        plt.plot(logger.losses['train'], label='Train')
        plt.plot(logger.losses['test'], label='Test')
        plt.title('Loss')
        plt.legend()

        plt.subplot(3,1,2)

        preds_fl = scaler_pv.inverse_transform(logger.attention_state.prediction[:200].reshape(-1, 1))
        gt_fl = np.round(scaler_pv.inverse_transform(logger.attention_state.label[:200].reshape(-1, 1)))

        plt.bar(range(200), preds_fl[:,0], alpha=0.5, label='Predicted', color='b')
        #plt.bar(range(200), np.round(preds_fl[:,0]), alpha=0.5, label='Rounded', color='g')
        plt.bar(range(200), gt_fl[:,0], alpha=0.5, label='Truth', color='r')
        plt.title('Sample predictions')
        plt.legend()

        plt.subplot(3,1,3)
        errs = np.subtract(gt_fl, preds_fl)
        plt.title('Error distribution (µ='+str(np.around(np.mean(errs),2))+ '; σ='+ str(np.around(np.std(errs),2))+')')
        plt.hist(errs)
        #plt.show()

        n = 10
        plt.figure(2,figsize=(10,n+1))
        attscaler = MinMaxScaler()
        for i in range(n):
            sequence = pd.DataFrame(logger.attention_state.inputs[i].tolist())
            sequence['attention'] = logger.attention_state.alphas[i][0].tolist()
            sequence['attention'] = attscaler.fit_transform(sequence['attention'].values.reshape(-1,1))

            plt.subplot(n,1,i+1)
            predval = np.round(preds_fl[i][0])
            gtval = np.round(gt_fl[i][0])
            plt.title('Attention map for sequence #'+str(i)+'; pred=' + str(int(predval))+'; gt='+str(int(gtval)))
            plt.imshow(sequence.transpose(), interpolation='nearest')
            plt.grid()
            plt.yticks([])
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

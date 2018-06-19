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
import seaborn as sns
plt.style.use('ggplot')

from ff_attention import FFAttention
from lstm_problems import lstm_problems
from utils import *

class PureFFAttention(torch.nn.Module):
    def __init__(self, batch_size, D_in, D_out, hidden):
        super(PureFFAttention, self).__init__()
        self.batch_size = batch_size
        self.n_features = D_in
        self.out_dim = D_out
        self.hidden = hidden

        # Layer 0a
        self.dropout0a = torch.nn.Dropout(p=0.2)
        self.layer0a = torch.nn.Linear(self.n_features, self.hidden)
        self.bn0a = torch.nn.BatchNorm1d(self.hidden)
        torch.nn.init.xavier_uniform_(self.layer0a.weight)
        # Layer 0b
        self.dropout0b = torch.nn.Dropout(p=0.2)
        self.layer0b = torch.nn.Linear(self.hidden, self.hidden)
        self.bn0b = torch.nn.BatchNorm1d(self.hidden)
        torch.nn.init.xavier_uniform_(self.layer0b.weight)
        # Layer 0b
        self.dropout0c = torch.nn.Dropout(p=0.2)
        self.layer0c = torch.nn.Linear(self.hidden, self.hidden)
        self.bn0c = torch.nn.BatchNorm1d(self.hidden)
        torch.nn.init.xavier_uniform_(self.layer0c.weight)

        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def forward(self, x, training=True):
        """
        Forward pass for the Feed Forward Attention network.
        """
        x = self.bn0a(F.leaky_relu(self.layer0a(x)))
        if self.training:
            x = self.dropout0a(x)
        x = self.bn0b(F.leaky_relu(self.layer0b(x)))
        if self.training:
            x = self.dropout0b(x)
        x = self.bn0c(F.leaky_relu(self.layer0c(x)))
        x = F.leaky_relu(self.out_layer(x))
        return x

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

def data_loader_pure(cache=False):
    # Cache config
    NPVS_CACHE = 'cache/npvs_ff_pure.dump'
    VP_CACHE = 'cache/vp_module_readouts_ff_pure.dump'
    SCALER_CACHE = 'cache/scaler_pv_ff_pure.dump'
    # Raw Bank Sizes (incl. number of PVs)
    npvs_file  = 'data/minbias10.npy'
    # Contains the occupancies for every VP sensor
    VP_file  = 'data/VP_occupancies_minbias10.npy'

    if cache == True:
        # Load PVs
        dfpv = pd.DataFrame(np.load(npvs_file))
        vp_occ = list(map(np.array, np.load(VP_file)))
        # Scale
        scaler_pv = MinMaxScaler()
        npvs = scaler_pv.fit_transform(np.asarray(dfpv['EventpRecVertexPrimary']).reshape(-1,1))
        joblib.dump(scaler_pv, SCALER_CACHE)

        # Load VP sensors
        dfvp = pd.DataFrame(dtype=float)
        for i in range(len(vp_occ[:-1])):
            dfvp[i] = list(map(float,vp_occ[i]))
        dfvp = dfvp.transpose()
        # Scale
        scaler_vp = MinMaxScaler()
        dfvp = scaler_vp.fit_transform(dfvp)
        vp_module_readouts = np.asarray(dfvp)

        # Cache data
        joblib.dump(npvs, NPVS_CACHE)
        joblib.dump(vp_module_readouts, VP_CACHE)
    else:
        npvs = joblib.load(NPVS_CACHE)
        vp_module_readouts = joblib.load(VP_CACHE)
        scaler_pv = joblib.load(SCALER_CACHE)

    return npvs, vp_module_readouts, scaler_pv

def plot_results(logger, scaler, show_results=True):
    if show_results is True:
        # Prepare predictions and Ground Truth for analysis
        preds_fl = scaler.inverse_transform(logger.attention_state.prediction.reshape(-1, 1))[:,0]
        gt_fl = np.round(scaler.inverse_transform(logger.attention_state.label.reshape(-1, 1)))[:,0]

        ############################################################
        # LOSS
        plt.figure(figsize=(15,10))
        plt.subplot(3,1,1)
        plot_loss(logger)
        # Sample Predictions
        plt.subplot(3,1,2)
        plot_predictions(y_true=gt_fl, y_pred=preds_fl, scaler=scaler_pv, sample=200)
        # Error distribution
        plt.subplot(3,1,3)
        plot_error(y_true=gt_fl, y_pred=preds_fl, rounded=True)

        ############################################################
        # CONFUSION
        plot_confusion(y_true=gt_fl, y_pred=preds_fl)

        plt.show()
        plt.tight_layout()

def main():
    """
    Run experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = AttentionLog()
    npvs, vp_module_readouts, scaler_pv = data_loader_pure()

    # Config
    load_model = False
    uid = '2018-06-19-13-44-PURE-FF'
    MODEL_PATH = 'vppv_pure_ff_model_best_epoch' + uid + '.pth'

    epoch_num = 100                     # Number of epochs to train the network
    batch_size = 100                    # Number of samples in each batch
    lr = 0.003                          # Learning rate
    n_seqs = 9000                       # number of sequences == number of samples
    D_in = vp_module_readouts.shape[1]  # 4 modules per sensor
    D_out = npvs.shape[1]               # Dimension of value to predict (1 here)
    D_hidden = 104                      # Hidden dimension

    batch_per_ep = n_seqs // batch_size # calculate the number of batches per epoch

    # Subsample data
    npvs = npvs[:n_seqs]
    vp_module_readouts = vp_module_readouts[:n_seqs,:]

    # Create or load the model
    model = PureFFAttention(batch_size=batch_size, D_in=D_in, D_out=D_out, hidden=D_hidden)
    if load_model is True:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['state_dict'])

    # Using the details from the paper [1] for optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999))
    if load_model is True:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tt_ratio = .75
    train_idx = np.random.uniform(0, 1, len(npvs)) <= tt_ratio

    # Define training data
    seq_ds = VPSequenceDataset(vp_module_readouts[train_idx], npvs[train_idx])
    # Define training data loader
    seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds,
                                                     batch_size=batch_size,
                                                     shuffle=True)

    # Define test data
    seq_ds_test = VPSequenceDataset(vp_module_readouts[~train_idx], npvs[~train_idx])
    # Define test data loader
    test_seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds_test,
                                                          batch_size=batch_size,
                                                          shuffle=True)

    preds = []; gt = [];
    for ep in range(epoch_num):  # epochs loop
        if load_model is True: # Skip training if pre-trained model is loaded
            break
        batch_losses = [];
        for batch_idx, data in enumerate(seq_dataset_loader): # batches loop
            features, labels = data

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(features, training=True)

            loss = criterion(output, labels.view(-1,1).float())
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
            output = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            label_log.append(labels.tolist())

            loss = criterion(output, labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

        if not ep % 5:
            print('[TEST] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep), 'loss: ', loss.data.item())

        logger.losses['test'].append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(logger.losses['test']):
            logger.best_epoch = ep
            logger.attention_state = AttentionState(alphas=[], inputs=features, label=flatten(label_log), prediction=flatten(outputs))
            save_state = {
            'epoch': ep,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }

            torch.save(save_state, MODEL_PATH)

    if load_model is True:
        batch_losses = []; outputs = []; label_log = [];
        for batch_idx, data in enumerate(test_seq_dataset_loader):
            features, labels = data
            output = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            label_log.append(labels.tolist())

            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())
        logger.losses['test'].append(np.mean(batch_losses))
        logger.attention_state = AttentionState(alphas=[], inputs=features, label=flatten(label_log), prediction=flatten(outputs))
    else:
        print("=== Best epoch:")
        print("Epoch #", logger.best_epoch)
        print("Test Loss = ", logger.losses['test'][logger.best_epoch])
        print("Train Loss = ", logger.losses['train'][logger.best_epoch])

    return logger

if __name__ == '__main__':
    logger = main()
    SCALER_CACHE = 'cache/scaler_pv.dump'
    scaler_pv = joblib.load(SCALER_CACHE)
    plot_results(logger, scaler_pv)

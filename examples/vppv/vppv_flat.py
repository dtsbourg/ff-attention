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
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from ff_attention import FFAttention
from lstm_problems import lstm_problems
from utils import *

class VPAttention(FFAttention):
    def __init__(self, *args, **kwargs):
        super(VPAttention, self).__init__(*args, **kwargs)
        ## Encoder
        # Layer 0a
        self.dropout0a = torch.nn.Dropout(p=0.2)
        self.layer0a = torch.nn.Linear(self.n_features, self.hidden)
        self.bn0a = torch.nn.BatchNorm1d(self.T)
        torch.nn.init.xavier_uniform_(self.layer0a.weight)
        # Layer 0b
        self.dropout0b = torch.nn.Dropout(p=0.2)
        self.layer0b = torch.nn.Linear(self.hidden, self.hidden)
        self.bn0b = torch.nn.BatchNorm1d(self.T)
        torch.nn.init.xavier_uniform_(self.layer0b.weight)
        # # Layer 0c
        # self.dropout0c = torch.nn.Dropout(p=0.2)
        # self.layer0c = torch.nn.Linear(self.hidden, self.hidden)
        # self.bn0c = torch.nn.BatchNorm1d(self.T)
        # torch.nn.init.xavier_uniform_(self.layer0c.weight)

        ## Decoder
        # Layer 1
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.layer1 = torch.nn.Linear(self.hidden, self.out_dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        # Out
        # # Layer a
        self.dropouta = torch.nn.Dropout(p=0.2)
        self.out_layera = torch.nn.Linear(self.hidden, self.hidden)
        self.bna = torch.nn.BatchNorm1d(self.out_dim)
        torch.nn.init.xavier_uniform_(self.out_layera.weight)
        # Layer b
        self.dropoutb = torch.nn.Dropout(p=0.2)
        self.out_layerb = torch.nn.Linear(self.hidden, self.hidden)
        self.bnb = torch.nn.BatchNorm1d(self.out_dim)
        torch.nn.init.xavier_uniform_(self.out_layerb.weight)

        # Layer c
        self.dropoutc = torch.nn.Dropout(p=0.2)
        self.out_layerc = torch.nn.Linear(self.hidden, self.hidden)
        self.bnc = torch.nn.BatchNorm1d(self.out_dim)
        torch.nn.init.xavier_uniform_(self.out_layerc.weight)

        # Layer d
        self.out_layerd = torch.nn.Linear(self.hidden, self.out_dim)
        self.bnd = torch.nn.BatchNorm1d(self.out_dim)
        torch.nn.init.xavier_uniform_(self.out_layerd.weight)

    def embedding(self, x_t):
        x_t = self.bn0a(F.leaky_relu(self.layer0a(x_t)))
        if self.training:
            x_t = self.dropout0a(x_t)
        x_t = self.bn0b(F.leaky_relu(self.layer0b(x_t)))
        if self.training:
            x_t = self.dropout0b(x_t)
        x_t = self.bn0c(F.leaky_relu(self.layer0c(x_t)))
        return x_t

    def activation(self, h_t):
        h_t = F.tanh(self.layer1(h_t))
        return h_t

    def out(self, c):
        c = self.bna(F.leaky_relu(self.out_layera(c)))
        if self.training:
            c = self.dropouta(c)
        c = self.bnb(F.leaky_relu(self.out_layerb(c)))
        if self.training:
            c = self.dropoutb(c)
        c = self.bnc(F.leaky_relu(self.out_layerc(c)))
        if self.training:
            c = self.dropoutc(c)
        c = self.bnd(F.leaky_relu(self.out_layerd(c)))
        return c

class VPSequenceDataset(Dataset):
    def __init__(self, vp_module_readouts, npvs):
        self.data = torch.from_numpy(vp_module_readouts).unsqueeze(-1)
        self.target = torch.from_numpy(npvs)

    def __getitem__(self, index):
        batch_labels = self.target[index]
        batch_data = self.data[index].float()
        return (batch_data, batch_labels)

    def __len__(self):
        return len(self.data)

def data_loader(cache=True):
    # Cache config
    NPVS_CACHE = 'cache/npvs.dump'
    VP_CACHE = 'cache/vp_module_readouts_flat.dump'
    SCALER_CACHE = 'cache/scaler_pv.dump'
    # Raw Bank Sizes (incl. number of PVs)
    npvs_file  = 'data/minbias10.npy'
    # Contains the occupancies for every VP sensor
    VP_file  = 'data/VP_occupancies_minbias10.npy'

    if cache == False:
        # Load PVs
        dfpv = pd.DataFrame(np.load(npvs_file))
        vp_occ = list(map(np.array, np.load(VP_file)))
        # Scale
        scaler_pv = StandardScaler()
        npvs = scaler_pv.fit_transform(np.asarray(dfpv['EventpRecVertexPrimary']).reshape(-1,1))
        joblib.dump(scaler_pv, SCALER_CACHE)

        # Load VP sensors
        dfvp = pd.DataFrame(dtype=float)
        for i in range(len(vp_occ[:-1])):
            dfvp[i] = list(map(float,vp_occ[i]))
        dfvp = dfvp.transpose()
        # Scale
        scaler_vp = StandardScaler()
        dfvp = scaler_vp.fit_transform(dfvp)
        #vp_module_readouts = np.asarray([module_seperator(_) for _ in dfvp])
        vp_module_readouts = np.asarray([_ for _ in dfvp])

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
        # ATTENTION
        plot_attention(logger, y_true=gt_fl, y_pred=preds_fl)

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
    npvs, vp_module_readouts, scaler_pv = data_loader(cache=True)

    # Config
    load_model = False
    uid = '2018-06-19-15-44'
    MODEL_PATH = 'models/vppv_model_best_epoch_flat' + uid + '.pth'

    epoch_num = 200                     # Number of epochs to train the network
    batch_size = 200                    # Number of samples in each batch
    lr = 0.001                          # Learning rate
    n_seqs = 9000                       # number of sequences == number of samples
    T = vp_module_readouts.shape[1]     # Sequence length == number of modules in our case
    D_in = 1
    D_out = npvs.shape[1]               # Dimension of value to predict (1 here)
    D_hidden = 104                      # Hidden dimension
    tt_ratio = .5
    train_idx = np.random.uniform(0, 1, len(npvs)) <= tt_ratio
    n_train = int(n_seqs * tt_ratio)
    n_test = int(n_seqs * (1-tt_ratio))

    # Create or load the model
    model = VPAttention(batch_size=batch_size, T=T, D_in=D_in, D_out=D_out, hidden=D_hidden)
    if load_model is True:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['state_dict'])

    # Using the details from the paper [1] for optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999), weight_decay=5e-4)

    if load_model is True:
        optimizer.load_state_dict(checkpoint['optimizer'])

    batch_per_ep = n_seqs // batch_size # calculate the number of batches per epoch
    batch_per_ep_tr = int(batch_per_ep * tt_ratio)
    batch_per_ep_te = batch_per_ep - batch_per_ep_tr

    # Define training data
    seq_ds = VPSequenceDataset(vp_module_readouts[train_idx][:n_train,:], npvs[train_idx][:n_train])
    # Define training data loader
    seq_dataset_loader = torch.utils.data.DataLoader(dataset=seq_ds,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Define test data
    seq_ds_test = VPSequenceDataset(vp_module_readouts[~train_idx][:n_test,:], npvs[~train_idx][:n_test])
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
            if batch_idx == batch_per_ep_tr - 1:
                break

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
            print('[TRAIN] epoch: {} --'.format(ep), 'loss: ', np.round(np.mean(batch_losses),3), '±', np.round(np.std(batch_losses),3))

        logger.losses['train'].append(np.mean(batch_losses))

        batch_losses = []; outputs = []; attention = []; label_log = [];
        for batch_idx, data in enumerate(test_seq_dataset_loader):
            if batch_idx == batch_per_ep_te - 1:
                break
            features, labels = data
            output, alphas = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            attention.append(alphas.tolist())
            label_log.append(labels.tolist())

            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

        if not ep % 10:
            print('[TEST]  epoch: {} --'.format(ep), 'loss: ', np.round(np.mean(batch_losses),3), '±', np.round(np.std(batch_losses), 3))

        logger.losses['test'].append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(logger.losses['test']):
            logger.best_epoch = ep
            logger.attention_state = AttentionState(alphas=alphas,
                                                    inputs=features,
                                                    label=flatten(label_log),
                                                    prediction=flatten(outputs))
            save_state = {
                'epoch': ep,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            torch.save(save_state, MODEL_PATH)

    if load_model is True:
        batch_losses = []; outputs = []; attention = []; label_log = [];
        for batch_idx, data in enumerate(test_seq_dataset_loader):
            features, labels = data
            output, alphas = model(features, training=False)

            outputs.append(flatten(output.tolist()))
            attention.append(alphas.tolist())
            label_log.append(labels.tolist())

            loss = criterion(output.view(-1,1), labels.view(-1,1).float())
            batch_losses.append(loss.data.item())

            if batch_idx == batch_per_ep_te - 1:
                break

        logger.losses['test'].append(np.mean(batch_losses))
        logger.attention_state = AttentionState(alphas=alphas,
                                                inputs=features,
                                                label=flatten(label_log),
                                                prediction=flatten(outputs))
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

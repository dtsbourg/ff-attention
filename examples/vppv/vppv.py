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

class VPAttention(FFAttention):
    def __init__(self, *args, **kwargs):
        super(VPAttention, self).__init__(*args, **kwargs)
        self.dropout0a = torch.nn.Dropout(p=0.2)
        self.layer0a = torch.nn.Linear(self.n_features, self.hidden)
        self.dropout0b = torch.nn.Dropout(p=0.2)
        self.layer0b = torch.nn.Linear(self.hidden, self.hidden)
        #self.dropout0c = torch.nn.Dropout(p=0.2)
        #self.layer0c = torch.nn.Linear(self.hidden, self.hidden)
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.layer1 = torch.nn.Linear(self.hidden, self.out_dim)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.layer2 = torch.nn.Linear(self.out_dim, self.hidden)
        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def embedding(self, x_t):
        x_t = F.leaky_relu(self.layer0a(x_t))
        if self.training:
            x_t = self.dropout0a(x_t)
        # x_t = F.leaky_relu(self.layer0b(x_t))
        # if self.training:
        #     x_t = self.dropout0b(x_t)
        #x_t = F.leaky_relu(self.layer0c(x_t))
        #if self.training:
        #    x_t = self.dropout0c(x_t)
        return x_t

    def activation(self, h_t):
        h_t = F.leaky_relu(self.layer1(h_t))
        if self.training:
            h_t = self.dropout1(h_t)
        return h_t

    def out(self, c):
        c = F.leaky_relu(self.layer2(c))
        if self.training:
            c = self.dropout2(c)
        return F.leaky_relu(self.out_layer(c))

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

def data_loader(cache=True):
    # Cache config
    NPVS_CACHE = 'cache/npvs.dump'
    VP_CACHE = 'cache/vp_module_readouts.dump'
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
        vp_module_readouts = np.asarray([module_seperator(_) for _ in dfvp])

        # Cache data
        joblib.dump(npvs, NPVS_CACHE)
        joblib.dump(vp_module_readouts, VP_CACHE)
    else:
        npvs = joblib.load(NPVS_CACHE)
        vp_module_readouts = joblib.load(VP_CACHE)
        scaler_pv = joblib.load(SCALER_CACHE)

    return npvs, vp_module_readouts, scaler_pv

def plot_results(logger, scaler_pv, show_results=True):
    if show_results is True:
        plt.style.use('ggplot')
        plt.figure(1,figsize=(15,10))
        plt.subplot(3,1,1)
        plt.plot(logger.losses['train'], label='Train')
        plt.plot(logger.losses['test'], label='Test')
        plt.title('Loss')
        plt.legend()

        plt.subplot(3,1,2)

        preds_fl = scaler_pv.inverse_transform(logger.attention_state.prediction.reshape(-1, 1))
        gt_fl = np.round(scaler_pv.inverse_transform(logger.attention_state.label.reshape(-1, 1)))

        sample = 200

        plt.bar(range(sample), preds_fl[:sample,0], alpha=0.5, label='Predicted', color='b')
        #plt.bar(range(200), np.round(preds_fl[:200,0]), alpha=0.5, label='Rounded', color='g')
        plt.bar(range(sample), gt_fl[:sample,0], alpha=0.5, label='Truth', color='r')
        plt.title('Sample predictions')
        plt.legend()

        plt.subplot(3,1,3)
        errs = np.subtract(gt_fl, preds_fl)
        errs_rounded = np.subtract(gt_fl, np.round(preds_fl))

        plt.hist(errs, label='Regression Error', alpha=0.5)
        plt.hist(errs_rounded, label='Rounded Error', alpha=0.5)
        mu_str = str(np.around(np.mean(errs),2)); std_str = str(np.around(np.std(errs),2));
        mu_rd_str = str(np.around(np.mean(errs_rounded),2)); std_rd_str = str(np.around(np.std(errs_rounded),2));
        plt.title('Error distribution ($\mu$='+mu_str+'; $\sigma$='+std_str+')'+'($\mu_{r}$='+mu_rd_str+'; $\sigma_{r}$='+std_rd_str+')')
        plt.legend()

        common_colorscale = False
        n = 10; start_idx = 10;
        fig = plt.figure(2,figsize=(10,n+1))
        cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
        images = []
        vmin = 1e40
        vmax = -1e40

        for i in range(n):
            sequence = pd.DataFrame(logger.attention_state.inputs[i+start_idx].tolist())
            attscaler = MinMaxScaler(feature_range=(np.min(np.min(sequence)),np.max(np.max(sequence))))
            sequence['attention'] = logger.attention_state.alphas[i+start_idx][0].tolist()
            sequence['attention'] = attscaler.fit_transform(sequence['attention'].values.reshape(-1,1))

            plt.subplot(n,1,i+1)

            predval = np.round(preds_fl[i+start_idx][0])
            gtval = np.round(gt_fl[i+start_idx][0])
            plt.title('Attention map for sequence #'+str(i+start_idx)+'; pred=' + str(int(predval))+'; gt='+str(int(gtval)))

            from numpy import amin, amax, ravel
            dd = ravel(sequence)
            # Manually find the min and max of all colors for
            # use in setting the color scale.
            vmin = min(vmin, amin(dd))
            vmax = max(vmax, amax(dd))
            images.append(plt.imshow(sequence.transpose(), interpolation='nearest'))

            plt.grid()
            plt.yticks([0,1,2,3,4], ['Sensor #1', 'Sensor #2', 'Sensor #3', 'Sensor #4', 'Attention'])
            plt.colorbar(aspect=5)

        if common_colorscale is True:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for i, im in enumerate(images):
                im.set_norm(norm)
                if i > 0:
                    images[0].callbacksSM.connect('changed', ImageFollower(im))

        plt.tight_layout()

        df = pd.DataFrame(list(zip(preds_fl[:,0],gt_fl[:,0])), columns=['preds', 'gt'])
        #sns.jointplot("preds", "gt", data=df, kind='kde')
        g = sns.JointGrid(x="preds", y="gt", data=df)
        g.plot_joint(sns.regplot, order=2)
        g.plot_marginals(sns.distplot)

        plt.tight_layout()
        plt.show()

class ImageFollower(object):
    'update image in response to changes in clim or cmap on another image'

    def __init__(self, follower):
        self.follower = follower

    def __call__(self, leader):
        self.follower.set_cmap(leader.get_cmap())
        self.follower.set_clim(leader.get_clim())

from matplotlib import cm, colors
from matplotlib.pyplot import sci

def main():
    """
    Run experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = AttentionLog()
    npvs, vp_module_readouts, scaler_pv = data_loader()

    # Config
    load_model = False
    uid = '2018-06-11-17-44-(104_2k5eps)_test_fix'
    MODEL_PATH = 'models/vppv_model_best_epoch' + uid + '.pth'

    epoch_num = 2                    # Number of epochs to train the network
    batch_size = 100                    # Number of samples in each batch
    lr = 0.001                          # Learning rate
    n_seqs = 1000                       # number of sequences == number of samples
    T = vp_module_readouts.shape[1]     # Sequence length == number of modules in our case
    D_in = vp_module_readouts.shape[2]  # 4 modules per sensor
    D_out = npvs.shape[1]               # Dimension of value to predict (1 here)
    D_hidden = 104                      # Hidden dimension

    tt_ratio = .5
    train_idx = np.random.uniform(0, 1, len(npvs)) <= tt_ratio
    n_train = int(n_seqs * tt_ratio)
    n_test = int(n_seqs * (1-tt_ratio))

    # Subsample data
    # npvs = npvs[:n_seqs]
    # vp_module_readouts = vp_module_readouts[:n_seqs,:]

    # Create or load the model
    model = VPAttention(batch_size=batch_size, T=T, D_in=D_in, D_out=D_out, hidden=D_hidden)
    if load_model is True:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['state_dict'])

    # Using the details from the paper [1] for optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999), weight_decay=1e-4)
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

        if not ep % 5:
            print('[TRAIN] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep_tr), 'loss: ', loss.data.item())

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

        if not ep % 5:
            print('[TEST] epoch: {} - batch: {}/{}'.format(ep, batch_idx, batch_per_ep_te), 'loss: ', loss.data.item())

        logger.losses['test'].append(np.mean(batch_losses))

        if np.mean(batch_losses) == np.min(logger.losses['test']):
            logger.best_epoch = ep
            logger.attention_state = AttentionState(alphas=alphas, inputs=features, label=flatten(label_log), prediction=flatten(outputs))
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
        logger.attention_state = AttentionState(alphas=alphas, inputs=features, label=flatten(label_log), prediction=flatten(outputs))

    return logger

if __name__ == '__main__':
    logger = main()
    SCALER_CACHE = 'cache/scaler_pv.dump'
    scaler_pv = joblib.load(SCALER_CACHE)
    plot_results(logger, scaler_pv)

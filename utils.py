#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ff-attention in Pytorch package, created by
# Dylan Bourgeois (@dtsbourg)
#
# License: MIT (see LICENSE.md)
#
# Utilities
#
# Logger:
# Contains some information about the state
import numpy as np

#######################
### Debug utilities ###
#######################
class AttentionState():
    def __init__(self, inputs, alphas, prediction, label, context_embedding=None):
        self.inputs = inputs
        self.alphas = alphas
        self.prediction = prediction
        self.label = label
        self.context_embedding = context_embedding

class AttentionLog():
    def __init__(self):
        self.losses = {'train': [], 'test': []}
        self.best_epoch = 0
        self.attention_state = None

flatten = lambda l: np.asarray([item for sublist in l for item in sublist])
module_seperator = lambda x: np.asarray([x[i:i+4] for i in range(0,len(x),4)])

######################
### Plot utilities ###
######################
class ImageFollower(object):
    'update image in response to changes in clim or cmap on another image'

    def __init__(self, follower):
        self.follower = follower

    def __call__(self, leader):
        self.follower.set_cmap(leader.get_cmap())
        self.follower.set_clim(leader.get_clim())

from matplotlib import cm, colors
from matplotlib.pyplot import sci
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from numpy import amin, amax, ravel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_loss(logger):
    plt.plot(logger.losses['train'], label='Train')
    plt.plot(logger.losses['test'], label='Test')
    plt.title('Loss')
    plt.legend()

def plot_predictions(y_true, y_pred, scaler=None, sample=200):
    # # WARNING: check index y_pred[:sample, 0] for VPPV
    plt.bar(range(sample), y_pred[:sample], alpha=0.5, label='Predicted', color='b')
    plt.bar(range(sample), y_true[:sample], alpha=0.5, label='Truth', color='r')
    plt.title('Sample predictions')
    plt.legend()

def plot_error(y_true, y_pred, rounded=False):
    errs = np.subtract(y_true, y_pred)
    errs_rounded = np.subtract(y_true, np.round(y_pred))

    plt.hist(errs, label='Regression Error', alpha=0.5)
    if rounded:
        plt.hist(errs_rounded, label='Rounded Error', alpha=0.5)
    mu_str = str(np.around(np.mean(errs),3)); std_str = str(np.around(np.std(errs),3))
    title_str = 'Error distribution ($\mu$='+mu_str+'; $\sigma$='+std_str+')'
    if rounded:
        mu_rd_str = str(np.around(np.mean(errs_rounded),3)); std_rd_str = str(np.around(np.std(errs_rounded),3));
        title_str = title_str+'($\mu_{o}$='+mu_rd_str+'; $\sigma_{o}$='+std_rd_str+')'
    plt.title(title_str)
    plt.legend()

def plot_attention(logger, y_true, y_pred, shared_colorscale=False, n_seq=8, start_idx=0):
    fig = plt.figure(figsize=(10,n_seq+1))
    images = []; vmin = 1e40; vmax = -1e40
    attn_dim = logger.attention_state.alphas.shape[2]
    data_dim = logger.attention_state.inputs.shape[2]
    for i in range(n_seq):
        sequence = pd.DataFrame(logger.attention_state.inputs[i+start_idx].tolist())
        attscaler = MinMaxScaler(feature_range=(np.min(np.min(sequence)),np.max(np.max(sequence))))
        for ad in range(attn_dim):
            att_idx = 'attention_{}'.format(ad)
            sequence[att_idx] = logger.attention_state.alphas[i+start_idx,:,ad].tolist()
            sequence[att_idx] = attscaler.fit_transform(sequence[att_idx].values.reshape(-1,1))

        plt.subplot(n_seq,1,i+1)

        predval = np.round(y_pred[i+start_idx],2)
        gtval = np.round(y_true[i+start_idx],2)
        plt.title('Attention map for sequence #'+str(i+start_idx)+'; pred=' + str(int(predval))+'; gt='+str(int(gtval)))

        dd = ravel(sequence)
        # Manually find the min and max of all colors for
        # use in setting the color scale.
        vmin = min(vmin, amin(dd)); vmax = max(vmax, amax(dd))
        images.append(plt.imshow(sequence.transpose(), interpolation='nearest'))

        plt.grid()
        data_str = ["Input #{}".format(_) for _ in range(data_dim)]
        attn_str = ["Attention #{}".format(_) for _ in range(attn_dim)]
        plt.yticks(range(attn_dim+data_dim), data_str+attn_str)
        plt.colorbar(aspect=5)

        if shared_colorscale:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for i, im in enumerate(images):
                im.set_norm(norm)
                if i > 0:
                    images[0].callbacksSM.connect('changed', ImageFollower(im))

def plot_confusion(y_true, y_pred):
    df = pd.DataFrame(list(zip(y_pred,y_true)), columns=['preds', 'gt'])
    #sns.jointplot("preds", "gt", data=df, kind='kde')
    g = sns.JointGrid(x="preds", y="gt", data=df)
    g.plot_joint(sns.regplot, order=2)
    g.plot_marginals(sns.distplot)

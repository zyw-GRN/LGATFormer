#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:31:24 2019

@author: lei.cai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as gnn
from torch.autograd import Variable
from utils import Evaluation

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False, dropout=0.04):
        super(Net, self).__init__()
        self.gatconv = gnn.GATConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.conv_params = nn.ModuleList()
        self.conv_params.append(self.gatconv(input_dim, latent_dim[0], heads=2, dropout=dropout))
        # self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.dropout = dropout
            # self.conv_params.append(conv(latent_dim[i-1], latent_dim[i], cached=False))
            self.conv_params.append(self.gatconv(latent_dim[i - 1]*2, latent_dim[i], heads=2))

        latent_dim = sum(latent_dim)*2
        self.encoder512 = nn.Linear(input_dim, 512)
        self.linear3 = nn.Linear(latent_dim+512, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear1 = nn.Linear(hidden_size[1], hidden_size[2])
        self.linear4 = nn.Linear(hidden_size[0], 2)
        
        self.with_dropout = with_dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.loss = nn.MSELoss()

    def forward(self, data):
        data.to(torch.device("cuda:0"))
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cur_message_layer = x
        cat_message_layers = []
        # cur_message_layer = cur_message_layer.float()
        #cat_message_layers.append(cur_message_layer)
        lv = 0
        cur_message_layer = cur_message_layer.float()
        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)
            # cur_message_layer = torch.tanh(cur_message_layer)
            cur_message_layer = F.elu(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        
        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch == i).nonzero()[0].cpu().numpy()[0])
        
        cur_message_layer = cur_message_layer[idx, :]
        chushi1 = x[idx, :]
        chushi = self.encoder512(chushi1)
        chushi = self.transformer_encoder(chushi)
        cur_message_layer = torch.cat((cur_message_layer, chushi), 1)

        # hidden1 = self.linear1(cur_message_layer)
        # self.feature = hidden1
        # hidden1 = F.relu(hidden1)
        # if self.with_dropout:
        #     hidden1 = F.dropout(hidden1, training=self.training)
        #
        # hidden1 = self.linear2(hidden1)
        # hidden1 = F.relu(hidden1)
        # if self.with_dropout:
        #     hidden1 = F.dropout(hidden1, training=self.training)

        #cur_message_layer = self.transformer_encoder(cur_message_layer)
        hidden = self.linear3(cur_message_layer)
        self.feature = hidden
        hidden = F.relu(hidden)
        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)

        # hidden = self.linear2(hidden)
        # self.feature = hidden
        # hidden = F.relu(hidden)
        # if self.with_dropout:
        #     hidden = F.dropout(hidden, training=self.training)
        # hidden = self.linear1(hidden)
        # self.feature = hidden
        # hidden = F.relu(hidden)
        # if self.with_dropout:
        #     hidden = F.dropout(hidden, training=self.training)

        logits = self.linear4(hidden)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, self.feature
        else:
            return logits
        
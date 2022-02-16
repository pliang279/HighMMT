#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 12:10:35 2018
@author: juanma
"""
# %%
import torch
import torch.nn as nn
import numpy as np


# %%
class SimpleRecurrentSurrogate(nn.Module):

    def __init__(self, num_hidden=100, number_input_feats=3, size_ebedding=100):
        # call the initialization method of super class
        # super(cls_name, self) convert to object of cls_name to its super class's object
        super(SimpleRecurrentSurrogate, self).__init__()

        self.num_hidden = num_hidden

        # input embedding
        self.embedding = nn.Sequential(nn.Linear(number_input_feats, size_ebedding),
                                       nn.Sigmoid())
        # the LSTM
        self.lstm = nn.LSTM(size_ebedding, num_hidden)
        # The linear layer that maps from hidden state space to output space
        self.hid2val = nn.Linear(num_hidden, 1)

        self.nonlinearity = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
                m.bias.data.fill_(1.8)

    def forward(self, sequence_of_operations):
        # (seq_len, batch, input_size):

        embeds = []
        for s in sequence_of_operations:
            embeds.append(self.embedding(s))
        embeds = torch.stack(embeds, dim=0)

        lstm_out, hidden = self.lstm(embeds)

        val_space = self.hid2val(lstm_out[-1])
        val_space = self.nonlinearity(val_space)

        return val_space

    def eval_model(self, sequence_of_operations_np, device):
        # the user will give this data sample as numpy array (int) with size len_seq x input_size

        npseq = np.expand_dims(sequence_of_operations_np, 1)
        sequence_of_operations = torch.from_numpy(npseq).float().to(device)
        res = self.forward(sequence_of_operations)
        res = res.cpu().detach().numpy()

        return res[0, 0]


# %%
class SurrogateDataloader():

    def __init__(self):
        self._dict_data = {}

    def add_datum(self, datum_conf, datum_acc):
        # data_conf is of size [seq_len, len_data]

        seq_len = len(datum_conf)
        datum_hash = datum_conf.data.tobytes()

        if seq_len in self._dict_data:

            if datum_hash in self._dict_data[seq_len]:
                # if the configuration is already stored, keep the max accuracy
                self._dict_data[seq_len][datum_hash] = (
                datum_conf, max(datum_acc, self._dict_data[seq_len][datum_hash][1]))
            else:
                self._dict_data[seq_len][datum_hash] = (datum_conf, datum_acc)
        else:
            self._dict_data[seq_len] = {datum_hash: (datum_conf, datum_acc)}

    def get_data(self, to_torch=False):
        # delivers list of numpy tensors of size [seq_len, num_layers, len_data]

        dataset_conf = list()
        dataset_acc = list()

        for len_key, data_dict in self._dict_data.items():

            conf_list = list()
            acc_list = list()

            for datum_hash, datum in data_dict.items():
                conf_list.append(datum[0])
                acc_list.append(datum[1])

            conf_list = np.transpose(np.asarray(conf_list, np.float32), (1, 0, 2))

            dataset_conf.append(np.array(conf_list, np.float32))
            dataset_acc.append(np.expand_dims(np.array(acc_list, np.float32), 1))

        if to_torch:
            for index in range(len(dataset_conf)):
                dataset_conf[index] = torch.from_numpy(dataset_conf[index])
                dataset_acc[index] = torch.from_numpy(dataset_acc[index])

        return dataset_conf, dataset_acc

    def get_k_best(self, k):

        dataset_conf = list()
        dataset_acc = list()

        for len_key, data_dict in self._dict_data.items():
            for datum_hash, datum in data_dict.items():
                dataset_conf.append(datum[0])
                dataset_acc.append(datum[1])

        dataset_acc = np.array(dataset_acc)
        top_k_idx = np.argpartition(dataset_acc, -k)[-k:]

        confs = [dataset_conf[i] for i in top_k_idx]
        accs = [dataset_acc[i] for i in top_k_idx]

        return (confs, accs, top_k_idx)


# %%
def train_simple_surrogate(model, criterion, optimizer, data_tensors, num_epochs, device):
    for epoch in range(num_epochs):

        model.train(True)  # Set model to training mode        

        # get the inputs
        for batch in range(len(data_tensors[0])):
            inputs, outputs = data_tensors[0][batch], data_tensors[1][batch]

            # move to device
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                f_outputs = model(inputs)

                loss = criterion(f_outputs, outputs)
                loss.backward()
                optimizer.step()

    model.train(False)
    return loss.item()

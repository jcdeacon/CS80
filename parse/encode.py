import torch

import sys
sys.path.insert(0, '../identity')
from data import *

def to_string(inp):
    hidden, cell = inp
    # The number of layers
    ret = str(len(hidden))
    ret += " +++$+++ "
    # The hidden layers, appended
    hidden = hidden.transpose(1,2)
    cell = cell.transpose(1,2)
    for i in hidden:
        for j in i:
            ret += str(float(j)) + " "
    ret += "+++$+++ "
    # The cell layers, appended
    for i in cell:
        for j in i:
            ret += str(float(j)) + " "
    ret = ret[:-1]
    return ret

def string_encode(datum, encoder, vocab):
    datum = prepare_string(datum)
    datum = ['SOS'] + datum + ['EOS']
    input_variable = variable_from_sentence(vocab, datum).transpose(0, 1)
    # get hidden states from encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    return to_string(encoder_hidden)

def from_string(hidden, num_layers):
    values = hidden.split(" ")
    length = len(values)
    for i in range(length):
        values[i] = float(values[i])

    layer_size = length // num_layers

    ls = []
    for i in range(num_layers):
        ls.append(values[i * layer_size: (i + 1) * layer_size])

    ret = torch.Tensor(ls)
    ret = ret.unsqueeze(1)
    return ret




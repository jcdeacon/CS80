import unicodedata
import string
import re
import random
import time
import math

import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

random_datum = True

def evaluate(sentence, total_length):
    input_variable = variable_from_sentence(vocab, sentence).transpose(0, 1)
    print("input variable length %d" % input_variable.size(0))
    input_length = input_variable.size()[1]

    # get hidden states from encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoded_words = []

    # TODO: allow other lengths

    # Run through decoder
    for di in range(input_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = int(topi[0][0])
        decoded_words.append(vocab.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
        if ni == EOS_token: break
    return decoded_words

def evaluate_randomly(bin_i):
    if random_from_test: datum = random.choice(test_data[bin_i])
    else: datum = random.choice(data[bin_i])

    output_words = evaluate(datum, bins[bin_i] + 2) # +2 for SOS and EOS
    output_sentence = ' '.join(output_words)

    print('>', datum)
    print('<', output_sentence)
    print('')

if random_datum:
    bin_i = 0

    total_length = bins[bin_i] + 2 # +2 for SOS and EOS.

    encoder = torch.load('encoder.pt')
    decoder = torch.load('decoder.pt')
    evaluate_randomly(bin_i)
else:
    bin_i = 0

    total_length = bins[bin_i] + 2 # +2 for SOS and EOS.

    encoder = torch.load('encoder.pt')
    decoder = torch.load('decoder.pt')

    words = input("Please enter a sentence: ")
    output_words = evaluate(words, total_length)
    output_sentence = ' '.join(output_words)

    print(output_sentence + "\n")


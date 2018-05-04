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

def test(sentence, total_length, encoder, decoder, max_length = MAX_LENGTH):
    input_variable = variable_from_sentence(vocab, sentence).transpose(0,1)
    input_length = input_variable.size(1)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    loss = 0 # Added onto for each word

    # TODO: allow for different lengths

    # Run through decoder
    for di in range(input_length):
        #import pdb; pdb.set_trace()
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        loss += criterion(decoder_output, input_variable[:,di])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    return loss/input_length

# Train!

######################################################################
######################################################################
######################################################################

teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, total_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # get hidden states from encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for _  in range(len(input_variable))]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # TODO: don't require that output havve the same size as the input.

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    for di in range(total_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, input_variable[:,di])

        if use_teacher_forcing:
            # Teacher forcing: Use the ground-truth target as the next input
            decoder_input = Variable(input_variable[:,di]) # Next target is next input
        else:
            # Without teacher forcing: use network's own prediction as the next input
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            # ezhan: you don't want to terminate sentence early during training,
            # otherwise the model might learn that the best way to decrease loss is to output EOS asap
            # if ni == EOS_token:
            #     break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / (total_length * batch_size)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

embedding_size = 10
hidden_size = 10
n_layers = 2

# Initialize models
encoder = EncoderRNN(vocab.n_words, embedding_size, hidden_size, n_layers)
decoder = DecoderRNN(vocab.n_words, embedding_size, hidden_size, n_layers)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# TODO: implement abilities for more bins.

bin_i = 0

#total_length = bins[bin_i]

total_length = 8
# Begin!
for epoch in range(1, n_epochs+1):
    if epoch % 500 == 0:
        print("On epoch %d" % epoch)
    # Get training data for this cycle
    '''input_variables = []
    for i in range(batch_size):
        input_variables.append(indexes_from_sentence(vocab, random.choice(data[bin_i])))
    input_variable = Variable(torch.LongTensor(input_variables).view(batch_size, -1, 1))'''
    #input_variable = variable_from_datum(random.choice(data[bin_i]))
    for batch_idx, data_batch in enumerate(train_dataloader):
        #print(batch_idx)
        if USE_CUDA:
            data_batch = data_batch.cuda()

        # Run the train function
        loss = train(data_batch, total_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        if epoch == 1 and batch_idx == 0:
            test_loss = []
            for i in range(len(test_data[bin_i])):
                testing_input = test_data[bin_i][i]
                test_loss.append(test(testing_input, total_length, encoder, decoder))
            prev_avg_test_loss = (sum(test_loss)/len(test_loss)).data[0]
            all_avg_test_loss = [int(prev_avg_test_loss)]
        if batch_idx == 1:
            test_loss = []
            for i in range(len(test_data[bin_i])):
                testing_input = test_data[bin_i][i]
                test_loss.append(test(testing_input, total_length, encoder, decoder))
            avg_test_loss = (sum(test_loss)/len(test_loss)).data[0]
            print("Average test loss:")
            print(avg_test_loss)
            all_avg_test_loss.append(float(avg_test_loss))
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            #if abs(prev_avg_test_loss - avg_test_loss) < convergence_value:
               #print("Average test losses:")
                #print(all_avg_test_loss)
                #break
            prev_avg_test_loss = avg_test_loss

            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(float(plot_loss_avg))
            plot_loss_total = 0
            torch.save(encoder, 'encoder.pt')
            torch.save(decoder, 'decoder.pt')
        if epoch % 2000 == 0:
            evaluate_randomly(bin_i)
    print("Examined %d data" % (epoch * len(data[0])))

torch.save(encoder, 'encoder.pt')
torch.save(decoder, 'decoder.pt')

print(plot_losses)
print(all_avg_test_loss)

show_plot(plot_losses)

evaluate_randomly(bin_i)


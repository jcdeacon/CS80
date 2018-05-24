import time
import math
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from model import *
from data import *
from evaluate import *

batch_size = 32

n_epochs = 15

vocab, (train_dataloader, test_dataloader), test_data  = prepare_dataloaders(batch_size)

# Train!

######################################################################
######################################################################
######################################################################

teacher_forcing_ratio = 0.25
clip = 5.0

def run(input_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word
    input_length = input_variable.size(1)

    # get hidden states from encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for _  in range(len(input_variable))]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if not train:
        use_teacher_forcing = False

    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        if (di >= input_length):
            real_value = torch.LongTensor([PAD_token for _ in range(len(input_variable))])
            if USE_CUDA:
                real_value = real_value.cuda()
        else:
            real_value = input_variable[:,di]
        loss += criterion(decoder_output, real_value)

        if use_teacher_forcing:
            # Teacher forcing: Use the ground-truth target as the next input
            decoder_input = Variable(real_value) # Next target is next input
        else:
            # Without teacher forcing: use network's own prediction as the next input
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

    # Backpropagation
    if train:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return float(loss) / MAX_LENGTH

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

if __name__ == '__main__':
    embedding_size = 100
    hidden_size = 200
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
    train_losses = []
    test_losses = []

    # Begin!
    for epoch in range(1, n_epochs+1):
        # Get training data for this cycle
        for batch_idx, data_batch in enumerate(train_dataloader):
            if USE_CUDA:
                data_batch = data_batch.cuda()

            # Run the train function
            loss = run(data_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, True)
            # Keep track of loss
            if batch_idx == 0 and epoch % 1 == 0:
                test_loss = []
                for bid, testing_input in enumerate(test_dataloader):
                    if USE_CUDA:
                        testing_input = testing_input.cuda()
                    test_loss.append(run(testing_input, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, False))
                avg_test_loss = sum(test_loss)/len(test_loss)
                test_losses.append(avg_test_loss)
                print_summary = '%s (%d %d%%) train: %.4f test: %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, loss, avg_test_loss)
                print(print_summary)
                train_losses.append(loss)
                plot_loss_total = 0
                torch.save(encoder, '../models/encoder-partial.pt')
                torch.save(decoder, '../models/decoder-partial.pt')

    torch.save(encoder, '../models/encoder.pt')
    torch.save(decoder, '../models/decoder.pt')

    now = datetime.datetime.now().strftime("_%B-%d-%Y_%H:%M")
    ep = str(n_epochs)
    torch.save(encoder, '../models/encoder_' + ep + now + '.pt')
    torch.save(decoder, '../models/decoder_' + ep + now + '.pt')

    print("Training loss:")
    print(train_losses)
    print("Testing loss:")
    print(test_losses)

    evaluate_randomly(test_data, encoder, decoder, vocab)


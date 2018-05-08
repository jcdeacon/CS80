import random

import torch
from torch.autograd import Variable

from data import *

random_datum = True

def evaluate(datum, encoder, decoder, vocab):
    input_variable = variable_from_sentence(vocab, datum).transpose(0, 1)
    # get hidden states from encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoded_words = []

    # Run through decoder
    for di in range(MAX_LENGTH):
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

def evaluate_randomly(test_data, encoder, decoder, vocab):
    datum = random.choice(test_data)

    output_words = evaluate(datum, encoder, decoder, vocab) # +2 for SOS and EOS
    output_sentence = ' '.join(output_words)

    print('>', datum)
    print('<', output_sentence)
    print('')

if __name__ == '__main__':
    encoder = torch.load('../models/encoder.pt')
    decoder = torch.load('../models/decoder.pt')
    vocab, (train_data, test_data) = read_data()

    if random_datum:
        evaluate_randomly(test_data, encoder, decoder, vocab)
    else:

        words = input("Please enter a sentence: ")
        words = prepare_string(words)
        words = ['SOS'] + words + ['EOS']
        output_words = evaluate(words, default_encoder, default_decoder, vocab)
        output_sentence = ' '.join(output_words)

        print(output_sentence + "\n")

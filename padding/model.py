import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)

    def forward(self, sentences):
        self.lstm.flatten_parameters() # In other places, this line errors
        batch_size = sentences.size(0)
        seq_len = sentences.size(1)

        h = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

        if sentences.is_cuda:
            h, c = h.cuda(), c.cuda()

        hidden = (h, c)

        embedded = self.embedding(sentences).transpose(0, 1)

        for t in range(seq_len):
            embedded_word = embedded[t].clone()
            output, hidden = self.lstm(embedded_word.unsqueeze(0), hidden)

        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.lstm.flatten_parameters()

    def forward(self, input, hidden):
        #self.lstm.flatten_parameters() TODO: figure out where to put this so it
        #doesn't modify important variables in place.
        output = self.embedding(input).squeeze(1)
        output, hidden = self.lstm(output.unsqueeze(0), hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

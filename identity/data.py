import re
import random
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

USE_CUDA = True

SOS_token = 0
EOS_token = 1

train_datafile = "../data/simple.txt"

MAX_LENGTH = 10

batch_size = 32

test_to_train = 0.1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

vocab = Lang("Script Vocab")

class LineDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = torch.LongTensor(datalist)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        return self.datalist[i]

def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def prepare_string(s):
    s = normalize_string(s)
    l = s.split(' ')
    return l

def prep(l):
    return ["SOS"] + l + ["EOS"]

def read_sentences(datafile):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(datafile).read().strip().split('\n')

    # normalize
    data = [prep(prepare_string(l)) for l in lines]

    return data

def prepare_data(datafile):
    data = read_sentences(datafile)
    print("Read %s sentences." % len(data))

    ret_data = ([],[]) # 0 is train, 1 is test

    print("Indexing words...")
    for datum in data:
        vocab.index_words(datum)
        # TODO: Remove the if when padding is implemented.
        if len(datum) == 8:
            if random.random() < test_to_train:
                ret_data[1].append(datum)
            else:
                ret_data[0].append(datum)

    print("Saving datasets...")

    with open('../data/trainset.pkl', 'wb') as f:
        pickle.dump(ret_data[0], f)

    with open('../data/testset.pkl', 'wb') as f:
        pickle.dump(ret_data[1], f)

    return ret_data

def read_data():
    with open('testset.pkl', 'rb') as f:
        test_data = pickle.load(f)

    with open('trainset.pkl', 'rb') as f:
        train_data = pickle.load(f)

    print("Trainset size: %d" % len(train_data))
    print("Testset size: %d" % len(test_data))

    return(train_data, test_data)

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def ready_for_dataset(datalist):
    ret = []
    for i in range(len(datalist)):
        ret.append(indexes_from_sentence(vocab, datalist[i]))
    return ret

def prepare_dataloaders():
    data = read_data()

    train_dataset = LineDataset(ready_for_dataset(data[0]))
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 8)
    test_dataset = LineDataset(ready_for_dataset(data[1]))
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers = 8)

    return (train_dataloader, test_dataloader)

def variable_from_sentence(vocab, sentence):
    indexes = indexes_from_sentence(vocab, sentence)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

if __name__ == '__main__':
    prepare_data(train_datafile)
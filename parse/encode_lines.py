from encode import *

f = open("../corpus/movie_lines.txt", 'r', errors = 'ignore')
f2 = open("../data/encoded_lines.txt", 'w')

encoder = torch.load('encoder.pt')
vocab, (_, _) = read_data()

i = 0
for l in f:
    if i % 1000 == 0:
        print(i)
    line = l
    line = line.split(" +++$+++ ")
    line[-1] = string_encode(line[-1], encoder, vocab)
    line = " +++$+++ ".join(line)
    f2.write(line)
    f2.write("\n")
    i += 1

f.close()
f2.close()


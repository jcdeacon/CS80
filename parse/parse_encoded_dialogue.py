import pickle

lines = open("../data/encoded_lines.txt", 'r')
dialogues = open("../corpus/movie_conversations.txt", 'r')

line_dict = {}
for line in lines:
    l = line.split(" +++$+++ ")
    line_dict[l[0]] = list(map(lambda x: float(x), l[-1].split(" ")))

print(line_dict['L195'])

data = []
for di in dialogues:
    datum = []
    d = di.split(" +++$+++ ")
    lst = d[-1][1:-2].split(", ")
    for i in lst:
        key = i[1:-1]
        datum.append(line_dict[key])
    data.append(datum)

with open("../data/dialogues.pkl", 'wb') as f:
    pickle.dump(data, f)

lines.close()
dialogues.close()

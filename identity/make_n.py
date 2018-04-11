def prep(string):
    for i in range(len(string)):
        if string[i] == ":":
            return string[i+2:]

n = 100
name = str(n) + ".txt"

with open("simple.txt", 'r') as f:
    with open(name, 'w') as f2:
        for i in range(n):
            f2.write(prep(f.readline()))
    with open("half_test_"+name, 'w') as f3:
        for i in range(n//2):
            f3.write(prep(f.readline()))


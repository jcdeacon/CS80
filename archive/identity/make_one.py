with open("simple.txt", 'r') as f:
    with open("one.txt", 'w') as f2:
        f2.write(f.readline())


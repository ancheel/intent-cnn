import random
import shutil
import os
import time




random.seed(time.time())


fp = open('data/paper.txt', 'w')
fc = open('data/conference.txt', 'w')
fe = open('data/expert.txt', 'w')

fpt = open('data/papert.txt', 'w')
fct = open('data/conferencet.txt', 'w')
fet = open('data/expertt.txt', 'w')

iter = 0
with open('data/intents.dat', 'r') as f:

    fl = f.readline()

    room = []
    while fl:

        word = fl.split('"')
        if word[7] in room:
            iter = iter + 1
            fl = f.readline()
            continue
        r = random.random()
        if word[3] == 'paper':
            if r > 0.1:
                fp.write(word[7] + '\n')
            else:
                fpt.write(word[7] + '\n')
        elif word[3] == 'expert':
            if r > 0.1:
                fe.write(word[7] + '\n')
            else:
                fet.write(word[7] + '\n')
        elif word[3] == 'conference':
            if r > 0.1:
                fc.write(word[7] + '\n')
            else:
                fct.write(word[7] + '\n')
        else:
            print(word[3])
        room.append(word[7])
        fl = f.readline()


print (iter)




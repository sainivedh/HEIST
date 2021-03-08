import numpy
from numpy import array
from numpy import argmax
from math import log
import torch




def beam_search_decoder(data, k):
    #print(data.shape)
    data = data.permute(1,0,2)
    data = torch.softmax(data, 2)
    data = data.detach().cpu().numpy()
    data = data[0]
    data = list(data)
    sequences = [[list(), 0.0]]

    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]

    return sequences
'''
#data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]

#data = array(data)

#result = beam_search_decoder(data, 3)

#for seq in result:
#    print(seq)


def remove_blanks(preds):
    out = []
    i = 0
    #print(len(preds))
    while (i < len(preds)):
        out.append(preds[i])
        temp = preds[i]
        print(temp)
        i = i + 1
        while (i < len(preds) and preds[i] == temp):
            i = i + 1
            #print(i)

    temp = "".join(out)
    print(temp.replace('§', ''))

arr = ['सा§§§§§व§§§§§ा§§§§म§§§§§§§']
remove_blanks(arr[0])
'''
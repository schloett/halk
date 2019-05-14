import networkx as nx
from math import sqrt, log, log2, ceil
import random
import time
import numpy as np


def shortest_path_pairs(G,src,tar, min_length):
    paths = dict()
    for cnt,a in enumerate(src):
        print(f'shortest path pairs for {cnt+1} of {len(src)} landmarks', end="\r")
        for b in tar:
            try:
                path = nx.shortest_path(G,a,b)
                length = 1
                for i in range(len(path)-1):
                    if length >= min_length:
                        paths[(path[0],path[i+1])] = length
                        paths[(path[i+1],path[0])] = length
                    length+=1
            except nx.NetworkXNoPath:
                pass
    return paths


def generate_data(edgelist, k_train=100, k_test=10, min_length=2):
    G = nx.read_edgelist(edgelist)
    nodes= list(G.nodes())

    # training
    landmarks_training = set()
    while len(landmarks_training) < k_train:
        landmarks_training.add(random.choice(nodes))
    #landmarks_training = [random.choice(nodes) for _ in range(k1)]
    remaining_nodes = set(nodes)-landmarks_training
    print(f'train set - all nodes {len(nodes)}, remaining nodes after selecting training landmarks {len(remaining_nodes)}')

    train = shortest_path_pairs(G,landmarks_training,remaining_nodes,min_length)

    print('test set')

    # test
    landmarks_test = set()
    while len(landmarks_test) < k_test:
        landmarks_test.add(random.choice(list(remaining_nodes)))
    remaining_nodes_test = remaining_nodes - landmarks_test

    test_all = shortest_path_pairs(G, landmarks_test, remaining_nodes_test,min_length)
    print('')
    test = {k:v for k,v in test_all.items() if k not in train}

    # min/max path length?
    print(f'{len(train)} training, {len(test)} test pairs')
	
    return train,test


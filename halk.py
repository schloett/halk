from collections import OrderedDict
from collections import defaultdict
from os import getcwd
import random
import numpy as np
from gensim.models import Word2Vec
from networkx.classes import DiGraph
from networkx.readwrite import edgelist


def reduce_walks(walks, keep_node_percentage):
    node_freq = defaultdict(int)
    for walk in walks:
        for node in walk:
            node_freq[node] += 1
    most_freq = sorted(node_freq, key=node_freq.get)[int((1 - keep_node_percentage) * len(node_freq)):]
    most_freq_nodes = {n: 1 for n in most_freq}
    reduced_walks = []
    for walk in walks:
        reduced_walk = [node for node in walk if node in most_freq_nodes]
        if reduced_walk:
            reduced_walks.append(reduced_walk)
    return {'reduced walks': reduced_walks, 'frequent nodes': [k for k in most_freq_nodes]}


def embed(walks, percentages, start_alphas, end_alphas, epochs, dimensions=128,workers=1,window=10,negative=5,min_count=0,sample=0.1, shuffle=True):
    walks = [[str(x) for x in walk] for walk in walks]
    model = Word2Vec(min_count=0, workers=workers, size=dimensions, window=window, sg=1, sample=0.1, negative=negative)
    model.build_vocab(walks)

    total_walk_entries = 0
    for step, percentage in enumerate(percentages):
        print(f'training {step + 1} of {len(percentages)}') 
        reduced_walks = reduce_walks(walks, percentage)
        if shuffle:
            random.shuffle(reduced_walks['reduced walks'])
        #print(len(reduced_walks['reduced walks']))
        #print(sum(len(i) for i in reduced_walks['reduced walks']))
        total_walk_entries += sum(len(i) for i in reduced_walks['reduced walks'])
        model.alpha = start_alphas[step]
        model.min_alpha = end_alphas[step]
        model.iter = epochs[step]
        model.train(reduced_walks['reduced walks'], total_examples=len(reduced_walks['reduced walks']))

    #print(f'{total_walk_entries} total walk entries')
    return model

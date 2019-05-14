import random

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import networkx as nx


def create_graph(file_name):
    """
    Reading an edge list csv and returning an Nx graph object.
    :param file_name: location of the csv file.
    :return graph: Nx graph object.
    """
    graph = nx.read_edgelist(file_name)
    return graph

def walk_transformer(walk, length):
    """
    Tranforming a given random walk to have skips.
    :param walk: Random walk as a list.
    :param length: Skip size.
    """
    transformed_walk = []
    for step in range(1,length+1):
        transformed_walk.append([y for i, y in enumerate(walk[step:]) if i % length ==0])
    return transformed_walk

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WalkletMachine:
    """
    Walklet multi-scale graph factorization machine class.
    The graph is being parsed up, random walks are initiated, embeddings are fitted, concatenated and the multi-scale embedding is dumped to disk.
    """

    def __init__(self, args):
        """
        Walklet machine constructor.
        :param args: Arguments object with the model hyperparameters. 
        """
        self.args = args
        self.graph = create_graph(self.args.input)
        self.walks = []
        #self.do_walks()
        #self.create_embedding()
        #self.save_model()

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        for step in range(self.args.walk_length - 1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        #walk = map(lambda x: str(x), walk)
        return walk

    def do_walks(self):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        """
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.args.walk_number):
            print("\nRandom walk round: " + str(iteration + 1) + "/" + str(self.args.walk_number) + ".\n")
            for node in tqdm(self.graph.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)

    def walk_extracts(self, length):
        """
        Extracted walks with skip equal to the length.
        :param length: Length of the skip to be used.
        :return good_walks: The attenuated random walks.
        """
        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_embedding(self, model):
        """
        Extracting the embedding according to node order from the embedding model.
        :param model: A Word2Vec model after model fitting.
        :return embedding: A numpy array with the embedding sorted by node IDs.
        """
        embedding = []
        for node in range(0, len(self.graph.nodes())):
            embedding.append(list(model[str(node)]))
        embedding = np.array(embedding)
        return embedding

    def create_embedding(self):
        """
        Creating a multi-scale embedding.
        """
        self.embedding = []

        for index in range(1, self.args.window_size + 1):
            print("\nOptimization round: " + str(index) + "/" + str(self.args.window_size) + ".")
            print("Creating documents.")
            clean_documents = self.walk_extracts(index)
            print("Fitting model.")
            empty = True
            for d in clean_documents:
                if len(d) > 0:
                    empty = False
            if not empty:
                model = Word2Vec(clean_documents,
                                 size=self.args.dimensions,
                                 window=self.args.window,
                                 min_count=self.args.min_count,
                                 sg=1,
                                 workers=self.args.workers,
                                 sample=self.args.sample,
                                 negative=self.args.negative,
                                 iter=self.args.iter,
                                 min_alpha=self.args.min_alpha,
                                 alpha=self.args.alpha
                                 )

                new_embedding = self.get_embedding(model)
                self.embedding = self.embedding + [new_embedding]
        self.embedding = np.concatenate(self.embedding, axis=1)

    def save_model(self):
        """
        Saving the embedding as a csv with sorted IDs.
        """
        print("\nModels are integrated to be multi scale.\nSaving to disk.")
       # self.column_names = map(lambda x: "x_" + str(x), range(self.embedding.shape[1]))
       # self.embedding = pd.DataFrame(self.embedding)
        conv = []
        for x in self.embedding:
            conv.append(np.array(x))
        self.embedding = PCA(n_components=self.args.dimensions).fit_transform(conv)
        self.save_w2v(self.embedding, self.args.output)
        #tuples = []
        #for i,emb in enumerate(self.embedding):
        #    tuples.append((i, np.array(emb)))
        #self.embedding = pd.DataFrame(tuples, columns=['node', 'embedding'])
        #self.embedding.to_pickle(self.args.output)

    def save_w2v(self, data, path):
        with open(path, 'w+') as f:
            f.write(str(len(data))+' '+str(len(data[0]))+'\n')
            for i, emb in enumerate(data):
                f.write(str(i))
                for e in emb:
                    f.write(' '+str(e))
                f.write('\n')





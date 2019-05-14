import os
import pickle

import networkx as nx
import numpy as np
from sklearn import metrics, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

default_params = {
    'edge_function': "hadamard",  # Default edge function to use
    "prop_pos": 0.5,  # Proportion of edges to remove nad use as positive samples
    "prop_neg": 0.5,  # Number of non-edges to use as negative samples
    #  (as a proportion of existing edges, same as prop_pos)
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


class Graph():
    def __init__(self,
                 nx_G=None, is_directed=False,
                 prop_pos=0.5, prop_neg=0.5,
                 workers=1,
                 random_seed=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.prop_pos = prop_neg
        self.prop_neg = prop_pos
        self.wvecs = None
        self.workers = workers
        self._rnd = np.random.RandomState(seed=random_seed)

    def read_graph(self, input, enforce_connectivity=True, weighted=False, directed=False):
        '''
        Reads the input network in networkx.
        '''
        #         if weighted:
        #             G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        #         else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G.adj[edge[0]][edge[1]]['weight'] = 1

        print("Read graph, nodes: %d, edges: %d" % (G.number_of_nodes(), G.number_of_edges()))

        G1 = G.to_undirected()
        self.G = G1

    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.
        Modify graph by removing the postive links.
        """
        # Select n edges at random (positive samples)
        n_edges = self.G.number_of_edges()
        n_nodes = self.G.number_of_nodes()
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        if not nx.is_connected(self.G):
            raise RuntimeError("Input graph is not connected")

        n_neighbors = [len(list(self.G.neighbors(v))) for v in self.G.nodes()]  ##
        n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        non_edges = [e for e in nx.non_edges(self.G)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        if len(neg_edge_list) < nneg:
            raise RuntimeWarning(
                "Only %d negative edges found" % (len(neg_edge_list))
            )

        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        edges = self.G.edges()

        edges = list(edges)

        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        rnd_inx = self._rnd.permutation(n_edges)

        for eii in rnd_inx.tolist():
            edge = edges[eii]

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)

            # Check if graph is still connected
            # TODO: We shouldn't be using a private function for bfs
            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                n_ignored_count += 1
            else:
                pos_edge_list.append(edge)
                print("Found: %d    " % (n_count), end="\r")
                n_count += 1

            # Exit if we've found npos nodes or we have gone through the whole list
            if n_count >= npos:
                print("")
                break

        if len(pos_edge_list) < npos:
            raise RuntimeWarning("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def edges_to_features(self, edge_list, edge_function, emb_size, model):
        """
        Given a list of edge lists and a list of labels, create
        an edge feature array using binary_edge_function and
        create a label array matching the label in the list to all
        edges in the corresponding edge list
        :param edge_function:
            Function of two arguments taking the node features and returning
            an edge feature of given dimension
        :param dimension:
            Size of returned edge feature vector, if None defaults to
            node feature size.
        :param k:
            Partition number. If None use all positive & negative edges
        :return:
            feature_vec (n, dimensions), label_vec (n)
        """
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, emb_size), dtype='f')

        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(model[str(v1)])
            emb2 = np.asarray(model[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec


def create_train_test_graphs(input='Facebook.edges', regen='regen', workers=8):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.
#     :param args:
#     :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # Create random training and test graphs with different random edge selections
    cached_fn = "%s.graph" % (os.path.basename(input))
    if os.path.exists(cached_fn) and not regen:
        print("Loading link prediction graphs from %s" % cached_fn)
        with open(cached_fn, 'rb') as f:
            cache_data = pickle.load(f)
        Gtrain = cache_data['g_train']
        Gtest = cache_data['g_test']

    else:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = Graph(is_directed=False,
                       prop_pos=prop_pos,
                       prop_neg=prop_neg,
                       workers=workers)
        Gtrain.read_graph(input)
        Gtrain.generate_pos_neg_links()

        # Generate a different random graph for testing
        Gtest = Graph(is_directed=False,
                      prop_pos=prop_pos,
                      prop_neg=prop_neg,
                      workers=workers)
        Gtest.read_graph(input)
        Gtest.generate_pos_neg_links()

        # Cache generated  graph
        #cache_data = {'g_train': Gtrain, 'g_test': Gtest}
        #with open(cached_fn, 'wb') as f:
        #    pickle.dump(cache_data, f)

    return Gtrain, Gtest


def test_edge_functions(Gtrain, Gtest, workers=8, num_experiments=1, emb_size=128, model=None,
                        edges_train=None, edges_test=None, labels_train=None, labels_test=None,lin_clf=LogisticRegression(C=1)):
    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    # TODO: It would be nice if the walks had a settable random seed
    aucs = {func: [] for func in edge_functions}
    for iter in range(num_experiments):

        for edge_fn_name, edge_fn in edge_functions.items():
            # Calculate edge embeddings using binary function
            edge_features_train = Gtrain.edges_to_features(edges_train, edge_fn, emb_size, model)
            edge_features_test = Gtest.edges_to_features(edges_test, edge_fn, emb_size, model)

            # Linear classifier
            scaler = StandardScaler()
            clf = pipeline.make_pipeline(scaler, lin_clf)

            # Train classifier
            clf.fit(edge_features_train, labels_train)
            auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

            # Test classifier
            auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)
            aucs[edge_fn_name].append(auc_test)

    return aucs


if __name__ == "__main__":  ###  'protein', 'karate', 'Facebook' 'blog',
    from dataset import read_df

    input = 'facebookRF.txt'
    Gtrain, Gtest = create_train_test_graphs(input, regen='regen', workers=8)

        # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    edges_test, labels_test = Gtest.get_selected_edges()


    ds = read_df('HALKLINK2')
    ds['node'] = ds['node'].map(lambda x: str(x))
    ds = zip(ds['node'], ds['embedding'])
    ds = dict(ds)


    functions = {"hadamard", "average", "l1", "l2"}

    auc = test_edge_functions(input, 'regen', workers=8, num_experiments=1, emb_size=128,
                              model=ds, edges_train=edges_train, edges_test=edges_test)

    with open('HALKLINK2result.txt', 'w+') as f:
      for i in functions:
        f.write(str(i) + '    AUC: ' + str('%.3f' % np.mean(auc[i])))
        f.write('\n')

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import getcwd

import numpy as np
import pandas as pd

import HP.graph_coarsening as graph_coarsening
import HP.magicgraph as magicgraph


def main():
    parser = ArgumentParser('harp',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--format', default='mat',
                        help='File format of input file')
    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')
    parser.add_argument('--sfdp-path', default='./bin/sfdp_osx',
                        help='Path to the SFDP binary file which produces graph coarsening results.')
    parser.add_argument('--model', default='deepwalk',
                        help='Embedding model to use. Could be deepwalk, line or node2vec.')
    parser.add_argument('--matfile-variable-name', default='network',
                        help='Variable name of adjacency matrix inside a .mat file')
    parser.add_argument('--number-walks', default=40, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--walk-length', default=10, type=int,
                        help='Length of the random walk started at each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of the Skip-gram model.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')
    args = parser.parse_args()

    # Process args
    if args.format == 'mat':
        G = magicgraph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=True)
    elif args.format == 'adjlist':
        G = magicgraph.load_adjacencylist(args.input, undirected=True)
    elif args.format == 'edgelist':
        G = magicgraph.load_edgelist(args.input, undirected=True)
    else:
        raise Exception("Unknown file format: '%s'. Valid formats: 'mat', 'adjlist', and 'edgelist'."
                        % args.format)
    G = graph_coarsening.DoubleWeightedDiGraph(G)
    print('Number of nodes: {}'.format(G.number_of_nodes()))
    print('Number of edges: {}'.format(G.number_of_edges()))
    print('Underlying network embedding model: {}'.format(args.model))

    if args.model == 'deepwalk':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=-1, iter_count=1,
                                                                       sfdp_path=args.sfdp_path,
                                                                       num_paths=args.number_walks,
                                                                       path_length=args.walk_length,
                                                                       representation_size=args.representation_size,
                                                                       window_size=args.window_size,
                                                                       lr_scheme='default', alpha=0.025,
                                                                       min_alpha=0.001, sg=1, hs=1, coarsening_scheme=2,
                                                                       sample=0.1)
    elif args.model == 'node2vec':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=-1, iter_count=1,
                                                                       sfdp_path=args.sfdp_path,
                                                                       num_paths=args.number_walks,
                                                                       path_length=args.walk_length,
                                                                       representation_size=args.representation_size,
                                                                       window_size=args.window_size,
                                                                       lr_scheme='default', alpha=0.025,
                                                                       min_alpha=0.001, sg=1, hs=0, coarsening_scheme=2,
                                                                       sample=0.1)
    elif args.model == 'line':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=1, iter_count=50,
                                                                       sfdp_path=args.sfdp_path,
                                                                       representation_size=64, window_size=1,
                                                                       lr_scheme='default', alpha=0.025,
                                                                       min_alpha=0.001, sg=1, hs=0, sample=0.001)
    np.save(args.output, embeddings)


def test_harp(input_file, output_file, sfdp):
    G = magicgraph.load_edgelist(input_file, undirected=True)
    # G = magicgraph.load_matfile(input_file, variable_name='network', undirected=True)
    G = graph_coarsening.DoubleWeightedDiGraph(G)
    print('Number of nodes: {}'.format(G.number_of_nodes()))
    print('Number of edges: {}'.format(G.number_of_edges()))
    embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=-1, iter_count=1,
                                                                   sfdp_path=sfdp,
                                                                   num_paths=10,
                                                                   path_length=80,
                                                                   representation_size=128,
                                                                   window_size=10,
                                                                   lr_scheme='default', alpha=0.025,
                                                                   min_alpha=0.001, sg=1, hs=0, coarsening_scheme=2,
                                                                   sample=0.1)
    np.save(output_file, embeddings)


def load_graph(file):
    G = magicgraph.load_edgelist(file, undirected=True)
    G = graph_coarsening.DoubleWeightedDiGraph(G)
    return G


def deepwalk(G, scale=-1, iter_count=1, sfdp_path=getcwd() + '/SFDP/sfdp_osx', num_walks=40, walk_length=10,
             representation_size=128, window_size=10, lr_scheme='default', alpha=0.025, min_alpha=0.001, sg=1, hs=1,
             coarsening_scheme=2, sample=0.1):
    return graph_coarsening.skipgram_coarsening_disconnected(G, scale=scale, iter_count=iter_count,
                                                             sfdp_path=sfdp_path,
                                                             num_paths=num_walks,
                                                             path_length=walk_length,
                                                             representation_size=representation_size,
                                                             window_size=window_size,
                                                             lr_scheme=lr_scheme, alpha=alpha,
                                                             min_alpha=min_alpha, sg=sg, hs=hs,
                                                             coarsening_scheme=coarsening_scheme,
                                                             sample=sample)

def dw(G, scale=-1, iter_count=1, sfdp_path=getcwd() + '/SFDP/sfdp_osx', num_walks=40, walk_length=10,
             representation_size=128, window_size=10, lr_scheme='default', alpha=0.025, min_alpha=0.001, sg=1, hs=1,
             coarsening_scheme=2, sample=0.1, outfile='emb.harp.wv',workers=1,negative=5,min_count=0):
    embeddings = graph_coarsening.skipgram_coarsening_disconnected(G, scale=scale, iter_count=iter_count,
                                                             sfdp_path=sfdp_path,
                                                             num_paths=num_walks,
                                                             path_length=walk_length,min_count=min_count,
                                                             representation_size=representation_size,
                                                             window_size=window_size,negative=negative,
                                                             lr_scheme=lr_scheme, alpha=alpha,
                                                             min_alpha=min_alpha, sg=sg, hs=hs,
                                                             coarsening_scheme=coarsening_scheme,
                                                             sample=sample,workers=workers)
    to_w2v(embeddings, outfile)


def to_w2v(data, path):
    with open(path, 'w+') as f:
        f.write(str(len(data)) + ' ' + str(len(data[0])) + '\n')
        for i, emb in enumerate(data):
            f.write(str(i))
            for e in emb:
                f.write(' ' + str(e))
            f.write('\n')

def line(G, scale=1, iter_count=50, sfdp_path=getcwd() + '/SFDP/sfdp_osx', representation_size=64, window_size=1,
         lr_scheme='default', alpha=0.025,
         min_alpha=0.001, sg=1, hs=0, sample=0.001):
    return graph_coarsening.skipgram_coarsening_disconnected(G, scale=scale, iter_count=iter_count,
                                                             sfdp_path=sfdp_path,
                                                             representation_size=representation_size,
                                                             window_size=window_size,
                                                             lr_scheme=lr_scheme, alpha=alpha,
                                                             min_alpha=min_alpha, sg=sg, hs=hs, sample=sample)


def create_df(embeddings):
    data = {'node': [i for i in range(embeddings.shape[0])], 'embedding': [row for row in embeddings]}
    df = pd.DataFrame(data=data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


if __name__ == '__main__':
   # n2v(load_graph('../facebookRF.txt'), iter_count=1, save_w2v=True, w2v_path='Youtube.walklets.wv', num_walks=80,
   #     walk_length=40)

    from dataset import save_df, append_label, create_ds_from_path
    df = n2v(load_graph('../files/blogcatalog/edges.txt'), iter_count=1, num_walks=80, walk_length=40)
    df = append_label(df, '../files/blogcatalog')
    from classification import classify_embeddings
    print(classify_embeddings(df, training_percentage=0.5, random_seed=3518955660))

    df = n2v(load_graph('../files/cora/edges.txt'), iter_count=1, num_walks=80, walk_length=40)
    df = append_label(df, '../files/cora')
    print(classify_embeddings(df, training_percentage=0.9, random_seed=3518955660))

    df = n2v(load_graph('../files/citeseer/edges.txt'), iter_count=1, num_walks=80, walk_length=40)
    df = append_label(df, '../files/citeseer')
    print(classify_embeddings(df, training_percentage=0.9, random_seed=3518955660))
    #from classification import classify_embeddings
    #print(classify_embeddings(df, training_percentage=0.9))

    #save_df(df, 'HARPLINK')
    # from HP.scoring import scoring
    # print(scoring('test.emb.npy', 'citeseer.mat', [0.9]))
    # G = magicgraph.load_matfile('citeseer.mat', undirected=True)

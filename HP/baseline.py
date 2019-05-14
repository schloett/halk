from gensim.models import Word2Vec

from HP import graph_coarsening


def skipgram_baseline(graph, **kwargs):
    scale = kwargs.get('scale', -1)
    representation_size = kwargs.get('representation_size', 128)

    if scale == 1:
        edges, weights = graph.get_edges()
    else:
        path_length = kwargs.get('path_length', 40)
        num_paths = kwargs.get('num_paths', 80)
        output = kwargs.get('output', 'default')
        edges = graph_coarsening.build_deepwalk_corpus(graph, num_paths, path_length, output)

    if kwargs['hs'] == 0:
        model = Word2Vec(edges, size=representation_size, window=kwargs['window_size'], min_count=kwargs['min_count'], min_alpha=kwargs['min_alpha'], alpha=kwargs['alpha'], sample=kwargs['sample'], sg=1, hs=0, iter=kwargs['iter_count'], negative=kwargs['negative'], workers=kwargs['workers'])
    else:
        model = Word2Vec(edges, size=kwargs['representation_size'], negative=kwargs['negative'], window=kwargs['window_size'], min_count=kwargs['min_count'], min_alpha=kwargs['min_alpha'], alpha=kwargs['alpha'], sample=kwargs['sample'], sg=1, hs=1, iter=kwargs['iter_count'], workers=kwargs['workers'])

    return model

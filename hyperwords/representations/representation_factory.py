from embedding import SVDEmbedding, EnsembleEmbedding, Embedding
from explicit import PositiveExplicit


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    neg = int(args['--neg'])
    w_c = args['--w+c']
    eig = float(args['--eig'])
    
    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return PositiveExplicit(path, True, neg)
    if rep_type == 'ICA':
        return PositiveExplicit(path, False, 1)
        
    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, False, eig, False), SVDEmbedding(path, False, eig, True), True)
        else:
            return SVDEmbedding(path, True, eig)
        
    else:
        if w_c:
            return EnsembleEmbedding(Embedding(path + '.words', False), Embedding(path + '.contexts', False), True)
        else:
            return Embedding(path + '.words', True)

### Performs iteration of word embeddings, leveraging context.

### For each word, builds a new representation by adding the 
### appended representations of their contexts stored in pairs_path.

### If input embeddings have size rep_size1 and we consider a context
### of size 2*win. the output embedding will have size
### rep_size2 = 2 * win * rep_size1

import numpy as np
from docopt import docopt
from representations.representation_factory import create_representation
from sklearn.decomposition import TruncatedSVD

def main():
    args = docopt("""
    Usage:
        rep2rep.py [options] <representation> <representation_path> <output_path> <pairs_path>
        
    Options:
        --win NUM   Window size for context to build new representation [default: 2]
        --new_dim NUM   Size of new (iterated) word vectors [default: 50]
        --neg NUM   Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c       Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM   Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
    """)

    output_path = args['<output_path>']
    
    aug_matrix = augment_representation(args)
    t_svd = TruncatedSVD(n_components = args['--new_dim'])
    iterated_rep = t_svd.fit_transform(aug_matrix)

    np.save(output_path, iterated_rep)

def augment_representation(args):
    """
    Creates a representation for words based on the words which appear
    as context to it.
    """

    representation = create_representation(args)
    pairs_path = args['<pairs_path>']
    win = int(args['--win'])

    old_dim = representation.dim
    temp_dim = 2 * win * old_dim

    # Initialize new (dense) matrix for word embeddings using contexts
    ctx_matrix = np.zeros([len(representation.wi), temp_dim])

    # Reads every word-pairs with positioning and adds context to extended word represenation
    with open(pairs_path, 'r') as fp:
        for line in fp:
            #word, ctx = line.strip().split('\t')
            word, ctx = line.strip().split()
            ctx_word, ctx_pos = ctx.rsplit('_', 1) # splits positional suffix
            ctx_pos = int(ctx_pos)
            ctx_vec = representation.represent(ctx) # gets context word vector

            if ctx_pos > 0:
                ctx_pos -= 1 # adjustment to map to array indexes

            range_start = (win + ctx_pos) * old_dim # where to add context
            range_end = range_start + old_dim
            # Add vector of context word to appropriate range of extended representation
            ctx_matrix[representation.wi[word], range_start:range_end] += ctx_vec

    return ctx_matrix
            
if __name__ == '__main__':
    main()

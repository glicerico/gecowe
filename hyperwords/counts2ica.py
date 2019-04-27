from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import FastICA
import numpy as np

from counts2pmi import read_counts_matrix
from representations.matrix_serializer import save_matrix, save_vocabulary, load_count_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2ica.py [options] <counts> <output_path>
    
    Options:
        --cps NUM    Number of ICA components to obtain [default: 50]
    """)
    
    counts_path = args['<counts>']
    vectors_path = args['<output_path>']
    
    counts, iw, ic = read_counts_matrix(counts_path)

    embeddings = calc_ica(counts, args['--cps'])

    save_matrix(vectors_path, embeddings)
    save_vocabulary(vectors_path + '.words.vocab', iw)
    save_vocabulary(vectors_path + '.contexts.vocab', ic)

def calc_ica(counts, cps):
    """
    Performs Independent Component Analysis (ICA) on counts 
    matrix, obtaining cps independent
    components, and returns dimension-reduced embeddings.
    """
    ica = FastICA(n_components = cps)
    embeddings = ica.fit_transform(counts.toarray())
    return csr_matrix(embeddings)

def calc_pmi(counts, cds):
    """
    Calculates e^PMI; PMI without the log().
    """
    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]
    if cds != 1:
        sum_c = sum_c ** cds
    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)
    
    pmi = csr_matrix(counts)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


if __name__ == '__main__':
    main()

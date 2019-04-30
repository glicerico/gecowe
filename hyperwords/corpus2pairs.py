from collections import Counter
from math import sqrt
from random import Random

from docopt import docopt


def main():
    args = docopt("""
    Writes to stdout each observation of a token-context pair.
    Sentences are padded to get fixed-size contexts (see param win below).
    In each observation, token and context are tab-separated

    Usage:
        corpus2pairs.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
        --pos        Positional contexts
        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --del        Delete out-of-vocabulary and subsampled placeholders
    """)
    
    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])
    pos = args['--pos']
    dyn = args['--dyn']
    subsample = float(args['--sub'])
    sub = subsample != 0
    d3l = args['--del']
    
    vocab = read_vocab(corpus_file, thr)
    corpus_size = sum(vocab.values())
    
    subsample *= corpus_size
    subsampler = dict([(word, 1 - sqrt(subsample / count)) for word, count in vocab.items() if count > subsample])
    
    filler = "#WALL#" # special token marking beginning/end of each line
    rnd = Random(17)
    with open(corpus_file) as f: 
        for line in f:
            
            tokens = [t if t in vocab else None for t in line.strip().split()]
            if sub:
                tokens = [t if t not in subsampler or rnd.random() > subsampler[t] else None for t in tokens]
            if d3l:
                tokens = [t for t in tokens if t is not None]
            
            len_tokens = len(tokens)
            
            # Pad lines with filler word
            padded_toks = [filler] * win
            temp_toks = tokens[:]
            temp_toks.extend(padded_toks)
            padded_toks.extend(temp_toks)
            for i, tok in enumerate(tokens):
                if tok is not None:
                    if dyn:
                        dynamic_window = rnd.randint(1, win)
                    else:
                        dynamic_window = win
                    start = i - dynamic_window
                    end = i + dynamic_window + 1
                    
                    if pos:
                        context = ', '.join([padded_toks[j + win] + '_' + str(j - i) for j in xrange(start, end) if j != i and padded_toks[j + win] is not None]).strip()
                    else:
                        context = ', '.join([padded_toks[j + win] for j in xrange(start, end) if j != i and padded_toks[j + win] is not None]).strip()
                    output = tok + '\t{' + context + '}'
                    if len(output) > 0:
                        print output


def read_vocab(corpus_file, thr):
    """
    Creates dictionary {token, count} for tokens appearing more than "thr" times
    """
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().split()))
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()

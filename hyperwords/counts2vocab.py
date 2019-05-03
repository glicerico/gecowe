from collections import Counter

from docopt import docopt

from representations.matrix_serializer import save_count_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2vocab.py [option] <counts>
    Options:
        --pos   Positional contexts
    """)
    
    counts_path = args['<counts>']
    pos = args['--pos']

    words = Counter()
    contexts = Counter()
    with open(counts_path) as f:
        for line in f:
            count, word, context = line.strip().split('\t')
            if pos:
                word, pos_suffix = word.rsplit('_', None, 1) # splits position suffix
            count = int(count)
            words[word] += count
            contexts[context] += count

    words = sorted(words.items(), key=lambda (x, y): y, reverse=True)
    contexts = sorted(contexts.items(), key=lambda (x, y): y, reverse=True)

    save_count_vocabulary(counts_path + '.words.vocab', words)
    save_count_vocabulary(counts_path + '.contexts.vocab', contexts)


if __name__ == '__main__':
    main()

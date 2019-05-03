#!/bin/sh

SOURCE_DIR=`dirname "$0"`

CORPUS=$1
# Clean the corpus from non alpha-numeric symbols
echo "Cleaning corpus..."
$SOURCE_DIR/scripts/clean_corpus.sh $CORPUS > $CORPUS.clean

# Create two example collections of word-context pairs:

# A) Window size 2 with "clean" subsampling
echo "Counting word-context pairs for win=2..."
mkdir -p w2.sub
python $SOURCE_DIR/hyperwords/corpus2pairs.py --thr 0 --win 2 ${CORPUS}.clean > w2.sub/pairs
python $SOURCE_DIR/hyperwords/corpus2pairs.py --thr 0 --win 2 --pos ${CORPUS}.clean > w2.sub/pairs_pos
$SOURCE_DIR/scripts/pairs2counts.sh w2.sub/pairs > w2.sub/counts
python $SOURCE_DIR/hyperwords/counts2vocab.py w2.sub/counts

# B) Window size 5 with dynamic contexts and "dirty" subsampling
# mkdir -p w5.dyn.sub.del
# python $SOURCE_DIR/hyperwords/corpus2pairs.py --win 5 --thr 0 --dyn --del ${CORPUS}.clean > w5.dyn.sub.del/pairs
# $SOURCE_DIR/scripts/pairs2counts.sh w5.dyn.sub.del/pairs > w5.dyn.sub.del/counts
# python $SOURCE_DIR/hyperwords/counts2vocab.py w5.dyn.sub.del/counts

# Calculate PMI matrices for each collection of pairs
echo "Calculating PMI matrices for word-context pairs for win=2..."
python $SOURCE_DIR/hyperwords/counts2pmi.py --cds 0.75 w2.sub/counts w2.sub/pmi
# python $SOURCE_DIR/hyperwords/counts2pmi.py --cds 0.75 w5.dyn.sub.del/counts w5.dyn.sub.del/pmi


# Create embeddings with SVD
echo "Creating SVD embeddings for word-context pairs for win=2..."
python $SOURCE_DIR/hyperwords/pmi2svd.py --dim 500 --neg 5 w2.sub/pmi w2.sub/svd
cp w2.sub/pmi.words.vocab w2.sub/svd.words.vocab
cp w2.sub/pmi.contexts.vocab w2.sub/svd.contexts.vocab
# python $SOURCE_DIR/hyperwords/pmi2svd.py --dim 500 --neg 5 w5.dyn.sub.del/pmi w5.dyn.sub.del/svd
# cp w5.dyn.sub.del/pmi.words.vocab w5.dyn.sub.del/svd.words.vocab
# cp w5.dyn.sub.del/pmi.contexts.vocab w5.dyn.sub.del/svd.contexts.vocab


# Evaluate on Word Similarity
echo
echo "WS353 Results"
echo "-------------"

python $SOURCE_DIR/hyperwords/ws_eval.py --neg 5 PPMI w2.sub/pmi $SOURCE_DIR/testsets/ws/ws353.txt
python $SOURCE_DIR/hyperwords/ws_eval.py --eig 0.5 SVD w2.sub/svd $SOURCE_DIR/testsets/ws/ws353.txt

# python $SOURCE_DIR/hyperwords/ws_eval.py --neg 5 PPMI w5.dyn.sub.del/pmi $SOURCE_DIR/testsets/ws/ws353.txt
# python $SOURCE_DIR/hyperwords/ws_eval.py --eig 0.5 SVD w5.dyn.sub.del/svd $SOURCE_DIR/testsets/ws/ws353.txt


# # Evaluate on Analogies
# echo
# echo "Google Analogy Results"
# echo "----------------------"

# python $SOURCE_DIR/hyperwords/analogy_eval.py PPMI w2.sub/pmi $SOURCE_DIR/testsets/analogy/google.txt
# python $SOURCE_DIR/hyperwords/analogy_eval.py --eig 0 SVD w2.sub/svd $SOURCE_DIR/testsets/analogy/google.txt

# python $SOURCE_DIR/hyperwords/analogy_eval.py PPMI w5.dyn.sub.del/pmi $SOURCE_DIR/testsets/analogy/google.txt
# python $SOURCE_DIR/hyperwords/analogy_eval.py --eig 0 SVD w5.dyn.sub.del/svd $SOURCE_DIR/testsets/analogy/google.txt

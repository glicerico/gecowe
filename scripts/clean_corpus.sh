#!/bin/sh

# Cleans raw corpus before counting word-context pairs
# 1) Converts any utf-8 characters to ascii
# 2) Converts text to lower caps
# 3) Converts to space non-alphanumeric chars from end or beginning of words, as well as any spacing within line
# 4) Converts to space non-alphanumeric chars at the end of each line
# 5) Compacts multiple spacing into one

# Usage clean_corpus.sh <raw_corpus>

iconv -c -f utf-8 -t ascii $1 | tr '[A-Z]' '[a-z]' | sed "s/[^a-z0-9]*[ \t\n\r][^a-z0-9]*/ /g" | sed "s/[^a-z0-9]*$/ /g" | sed "s/  */ /g"
#!/bin/sh

# Taking a file with word-context pairs, prints a count
# of each pair.
# Uses tab to separate the count from the pair

# Usage: pairs2counts.sh <pairs file>

sort -T . $1 | uniq -c | sed -E 's/^ *//; s/ /\t/'
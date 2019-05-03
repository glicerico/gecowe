### Performs iteration of word embeddings, leveraging context.

### For each word, builds a new representation by adding the 
### appended representations of their contexts stored in pairs_path.

### If input embeddings have size rep_size1 and we consider a context
### of size 2*win. the output embedding will have size
### rep_size2 = 2 * win * rep_size1

from docopt import docopt
from representations.representation_factory import create_representation

def main():
	args = docopt("""
	Usage:
		rep2rep.py [options] <representation> <representation_path> <output_path> <pairs_path>

	Options:
		--win NUM	Window size for context to build new representation [default: 2]
	""")

	output_rep = args['<output_rep>']
	pairs_path = args['<pairs_path>']
	win = int(args['--win'])
	
	representation = create_representation(args)



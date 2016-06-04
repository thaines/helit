Corpus

Builds and provides access to a corpus of short 'blocks' of text extracted from a large number of English language books. Its main value is in generating text for an author to write out, so a model of their handwriting can be learnt, as used by the paper 'My Text in Your Handwriting'. It is done this way so the authors would be given recognisable blocks of text, as we found that that asking a person to write something unusual results in them pausing a lot, which makes their handwriting unnatural. Note that the presented code introduces randomness, as required for our user tests - may not be otherwise desirable.

To use this library some preparation is required:


1. Get the data. It needs the folder 'data' to contain the relevant text file versions of the books used. Specifically, the top 100 books were downloaded from project Gutenberg (https://www.gutenberg.org/), at the time I created this (almost certainly different when you are reading this). The files are listed in make_db.py.

For convenience a zip file that can be unzipped into this directory (make sure to preserve directory structure) is available from:
http://thaines.com/content/research/2016_tog/corpus_data.zip

Also includes the files I did not ultimately use due to being work unsafe, containing foreign languages, or textualised accents.


2. Run python make_db.py. This will generate the corpus.ply2 database file that the other scripts use. Also outputs corpus.txt, so you can verify the contents is sensible.


Once prepared, there are three further scripts you may run, or you may include the Corpus object from corpus.py to access the data, and its statistics, for your own purposes. The scripts are:

test_corpus.py - Quick test script, that prints out some statistics and a random block of text from the corpus.
make_sample.py - Generates text for an author to write out, so you can capture their handwriting style.
make_samples.sh - Convenience bash script that runs the above many times, generating a file for each run, so you can walk away and come back to a directory full of texts to use.


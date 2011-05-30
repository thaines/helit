A variational LDA implementation, using the global variational method.

Fairly standard really, though it should be noted that it uses a different solving method to the original LDA paper ('Latent Dirichlet Allocation', by D. M. Blei, A. Y. Ng and M. I. Jordan.). It also has a proper Dirichlet prior over beta, to smooth in unseen words.

Implementation is all in straight python, so not as fast as it could be, but still pretty decent due to vectorisation. Makes use of scipy, and both tests use open cv. Unlike the Gibbs implementation it does not support multiprocessing, it also does not have progress bar support, but then its a lot faster, so these features are not as necessary.


The graphical model includes the current variables:

alpha - Known prior on theta, a Dirichlet distribution.
theta_d - The multinomial over topics for each document. Has a prior alpha; and the z values are drawn from this.
z_dn - Label assigned to each word, indicating which topic it was drawn from. Drawn from the documents theta, indexes the beta from which w is drawn.
w_dn - The known words, i.e. the data that is actually provided to the algorithm. Many of these for each document. Works are drawn from the beta distribution associated with the topic for that word, which is drawn from the documents theta.
beta_t - One for each topic - a multinomial from which words are drawn from.
gamma - Fixed prior over beta, a Dirichlet distribution.

d - subscript for document.
t - subscript for topic.
n - subscript for word within a document.


The distribution can be written:

P(theta,z,w,beta) = P(theta;alpha) P(z|theta) P(w|z,beta) P(beta;gamma)

and the variational factorisation approximation is:

q(theta,z,beta) = q(theta) q(z) q(beta)


The files included are:

lda.py - Everything basically - implementation and interface. Its a lot simpler than the Gibbs version!

test_grid.py - A test program.
test_junction.py - Another test program.

readme.txt - This file, which is copied into lda_var.html when generated.
make_doc.py - Generates the documentation.


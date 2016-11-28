"""
Save the word2vec vectors of the all the words in vocabulary.
If the word is not in word2vec, use the average embedding.
"""

import gensim
import numpy as np
import h5py

# Use gesim package to load word2vec
model = gensim.models.Word2Vec.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin.gz', binary=True)

# Load the vocabulary, add end of sentence and a avgvec (average vector) to the vocabulary.
vocab = open('data/vocabulary.txt','r').readlines() + ['</s>', 'avgvec']

# Get 
avg = np.mean(model.syn0, 0)

result = np.zeros((len(vocab), 300))
for i in range(len(vocab)):
	if vocab[i].strip() in model: # the word is in word2vec
		result[i,:] = model[vocab[i].strip()]
	else: # not in word2vec
		print vocab[i].strip()
		result[i, :] = avg

# Save the matrix into a h5 file
filename = 'data/w2v.h5'
f = h5py.File(filename, 'w')
f.create_dataset("w2v", dtype='float32', data=result)
f.close()


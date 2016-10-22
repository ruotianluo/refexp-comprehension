import gensim
import numpy as np
import h5py

model = gensim.models.Word2Vec.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin.gz', binary=True)

vocab = open('data/vocabulary.txt','r').readlines() + ['</s>', 'avgvec']

unk = np.mean(model.syn0, 0)

result = np.zeros((len(vocab), 300))
for i in range(len(vocab)):
	if vocab[i].strip() in model:
		result[i,:] = model[vocab[i].strip()]
	else:
		print vocab[i].strip()
		result[i, :] = unk

filename = 'data/w2v.h5'
f = h5py.File(filename, 'w')
f.create_dataset("w2v", dtype='float32', data=result)
f.close()


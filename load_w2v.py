import gensim
import numpy as np

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

vocab = open('xxx/vocabulary.txt','r').readlines() + ['bos', '</s>']:


unk = np.mean(model.syn0, 0)

result = np.zeros(len(vocab), )
for i in range(len(vocab)):
	if vocab[i].strip() in model:
		result[i,:] = model[vocab[i].strip()]
	else:
		print vocab[i].strip()
		result[i, :] = unk

np.save('w2v.npy',result)


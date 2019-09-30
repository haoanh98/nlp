import os
import numpy as np
embeddings_index = {}
f = open(os.path.join('', 'fb_embedding_word2vec.txt'),  encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()
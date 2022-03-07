# Small model (86MB): https://www.kaggle.com/murats/word2vec-application-on-turkish-newspaper/notebook
# Big model (633MB): https://github.com/akoksal/Turkish-Word2Vec

import time
from gensim.models import Word2Vec

# Small model
# 2.3576974868774414 seconds
# 0.7495866
start_time = time.time()
model_small = Word2Vec.load("word2vec_small")
test1 = model_small.wv.similarity("kral","prens")
print("Small model: %s seconds" % (time.time() - start_time))
print(test1)

# Big model
# 4.888732433319092 seconds
# 0.58733094
start_time = time.time()
model_big = Word2Vec().wv.load_word2vec_format("word2vec_big", binary=True)
test2 = model_big.similarity("kral","prens")
print("Big model: %s seconds" % (time.time() - start_time))
print(test2)
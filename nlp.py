import pandas as pd
df = pd.read_csv("./data.txt",sep="/",names=["row"]).dropna()
df.head(7)
import re
def transform_row(row):
    row = re.sub(r"^[0-9\.,]+", "", row) #
    
    row = re.sub(r"[\.,\?]+$", "", row)
    
    row = row.replace(",","").replace(".", "") \
        .replace(";", "").replace("“", "") \
        .replace(":", "").replace("”", "") \
        .replace('"', "").replace("'", "") \
        .replace("!", "").replace("?", "")
    row = row.strip()
    return row 

df["row"] = df.row.apply(transform_row)
df.head(10)

from nltk import ngrams

def kieu_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ " ".join(gram).lower() for gram in gram_str ]

df["1gram"] = df.row.apply(lambda t: kieu_ngram(t, 1))
df["2gram"] = df.row.apply(lambda t: kieu_ngram(t, 2))

df.head(10)

df["context"] = df["1gram"] + df["2gram"]
train_data = df.context.tolist()
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
model = Word2Vec(train_data, size=100, window=10, min_count=3, workers=4, sg=1)


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
total_status = df['row']
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(total_status)
sequences = tokenizer_obj.texts_to_sequences(total_status)
status_pad = pad_sequences(sequences, maxlen=8)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import sys
#import importlib
#importlib.reload(sys)
#sys.setdefaultencoding("utf-8")
words_np = []
words_label = []
for word in model.wv.vocab.keys():
    words_np.append(model.wv[word])
    words_label.append(word)

pca = PCA(n_components=2) 
pca.fit(words_np)
reduced = pca.transform(words_np)

plt.rcParams["figure.figsize"] = (20,20)
for index,vec in enumerate(reduced):
    if index <200:
        x,y=vec[0],vec[1]
        plt.scatter(x,y)
        plt.annotate(words_label[index],xy=(x,y))
plt.show()

    
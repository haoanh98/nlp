# Natural Language Processing
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from keras.optimizers import SGD
from keras.utils import to_categorical
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('bbc-text.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 2225):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    corpus.append(review)


from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
model = Word2Vec(corpus, size=100, window=10, min_count=3, workers=4, sg=1)

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)
word_index = tokenizer_obj.word_index
status_pad = pad_sequences(sequences, maxlen=200)


EMBEDDING_DIM =100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word in model.wv.vocab.keys():
    i=word_index[word]
    embedding_matrix[i] = model.wv[word]
    

y=[]
for word in dataset.iloc[:,0].values:
    if word=='business' :
        y.append([1 ,0 ,0 ,0 ,0 ])
    if word=='entertainment' :
        y.append([0 ,1 ,0 ,0 ,0 ])
    if word=='politics' :
        y.append([0 ,0 ,1 ,0 ,0 ])
    if word=='sport' :
        y.append([0 ,0 ,0 ,1 ,0 ])
    if word=='tech' :
        y.append([0 ,0 ,0 ,0 ,1 ])
y = np.asarray(y)        


            
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(status_pad, y, test_size = 0.20, random_state = 0)




model1 = Sequential()
# load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=200,
                            trainable=True)

model1.add(embedding_layer)
model1.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Flatten())
model1.add(Dense(10, input_dim=1, activation='relu'))
model1.add(Dense(5, activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
history=model1.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=20,batch_size=64,verbose=2)
print(model1.summary())
plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
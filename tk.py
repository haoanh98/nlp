import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bbc-text.csv')

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
from collections import Counter
df = Counter()
for i in range(0,2225):
    for word in corpus[i]:
        df[word]=df[word]+1
a=df.most_common(10)
x=[]
y=[]
for i in range(0,10):
    x.append(a[i][0])
    y.append(a[i][1])
plt.bar(x,y,color='green')
plt.xlabel('word')
plt.ylabel('times')
plt.show()


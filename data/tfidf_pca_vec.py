# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:13:33 2018

@author: 51645
"""

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
#改成你们的路径
df = pd.read_csv('fake_or_real_news.csv')

import nltk
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')

def clean_text(tokenized_list, sw):
    new_list = []
    nodes = [',', '*', '"', '.', '\'', '“', '”', '’', '‘', '，']
    for doc in tokenized_list:
        new_list.append([token.lower() for token in doc if token.lower() not in sw and token.lower() not in nodes])
    return new_list

texts = df.text

#create mapping for string translate method
mapping_table = {ord(char): u' ' for char in punctuation}

tokenized = [nltk.word_tokenize(review.translate(mapping_table)) for review in texts]

# Remove punctuations and stopwords, and lower-case text
sw = stopwords.words('english')
cleaned = clean_text(tokenized, sw)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

Tf_fit = []
for article in cleaned:
    str = ''
    for i in article:
        str += i
        str += ' '
    Tf_fit.append(str)
    
vectorizer.fit(Tf_fit)
tfidf_matrix = vectorizer.transform(Tf_fit).toarray()
print(tfidf_matrix)
print(len(tfidf_matrix[0]))

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pca = PCA(n_components = 700)
pca.fit(tfidf_matrix[:2500])

a = pca.transform(tfidf_matrix)
#改成你们自己的保存路径
a.tofile('tfidf_matrix_2500_700.bin')
#b = np.fromfile('C:\\Users\\51645\\Desktop\\fake_news\\')
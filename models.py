import pandas as pd
pd.set_option('display.max_colwidth', 150)
import keras.backend as K
import string
import nltk
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
#from keras_preprocessing.text import Tokenizer
#from keras_preprocessing.sequence import pad_sequences
pd.set_option('display.max_colwidth', 150)


# Load Our Data
X_train = pd.read_csv("Data/X_train.csv")
X_test = pd.read_csv("Data/X_test.csv")
y_train = pd.read_csv("Data/y_train.csv")
y_test = pd.read_csv("Data/y_test.csv")

print(X_train.head(5))
print(X_test.head(5))
# Let's create a TF-IDF Vectors

tfidf_vectors = TfidfVectorizer()
tfidf_vectors.fit(X_train['cat_var_tokenized'])
X_train_vectors = tfidf_vectors.transform(X_train['cat_var_tokenized'])
X_test_vectors = tfidf_vectors.transform(X_test['cat_var_tokenized'])

# Let's see what word did the vectorizer leraned

print(tfidf_vectors.vocabulary_)


# Show the TF_IDF Matrix
print(X_train_vectors.toarray())

# Show TF_IDF : We have 39 Categorical Variable
print(tfidf_vectors.get_feature_names())

# View Feature Matrix As Dataframe

print(pd.DataFrame(X_train_vectors.toarray(), columns=tfidf_vectors.get_feature_names()))
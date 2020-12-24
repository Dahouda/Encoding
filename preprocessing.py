import pandas as pd
import corpus_creation
import nltk
import re
from sklearn.model_selection import train_test_split

stopwords = nltk.corpus.stopwords.words('english')

# Create a function to Tokenize all the Categorical Data


def tokenize_cat_var(catvar):
    tokens = re.split('\W+', catvar)
    return tokens


corpus_creation.corpus['cat_var_tokenized'] = corpus_creation.corpus['cat_var'].apply(lambda x: tokenize_cat_var(x.lower()))
print(corpus_creation.corpus.head(5))


# Split the Data subset into train and test set
X_train, X_test, y_train, y_test = train_test_split(corpus_creation.corpus['cat_var_tokenized'], corpus_creation.corpus['subscribe'], test_size=0.2)

# Let's save the training and test sets to ensure we are using the same data for each model

X_train.to_csv("./Data/X_train.csv", index=False, header=True)
X_test.to_csv("./Data/X_test.csv", index=False, header=True)
y_train.to_csv("./Data/y_train.csv", index=False, header=True)
y_test.to_csv("./Data/y_test.csv", index=False, header=True)

# Let's see our Tokenized Categorical Variable
print("****Let's see our Tokenized Categorical Variable****")
X_train = pd.read_csv("Data/X_train.csv")
X_test = pd.read_csv("Data/X_test.csv")
y_train = pd.read_csv("Data/y_train.csv")
y_test = pd.read_csv("Data/y_test.csv")

print(X_train.head(5))
import pandas as pd
import numpy as np
import data_loading
import time
pd.set_option('display.max_colwidth', 150)

# How long takes The program to run
start_time = time.time()

# Import The Data Subset
data_loading.numeric_data.head()
data_loading.categorical_data.head()

# What is the shape of the Categorical Subset
print("Input Data has {} rows and {} columns".format(len(data_loading.categorical_data),
                                                     len(data_loading.categorical_data.columns)))

# How many yes / no are there ?
print("Out of {} rows, are yes, {} are no".format(len(data_loading.categorical_data),
                                                  len(data_loading.categorical_data[data_loading.categorical_data['deposit'] == 'yes']),
                                                  len(data_loading.categorical_data[data_loading.categorical_data['deposit'] == 'no'])))

# How much missing data is there ?
print("Number of null in deposit: {}".format(data_loading.categorical_data['deposit'].isnull().sum()))
print("Number of null in categorical variable:\n{}".format(data_loading.categorical_data.isnull().sum()))
print("Number of null in numeric variable: \n{}".format(data_loading.numeric_data.isnull().sum()))

# save the Categorical variable Data Subset into csv file
data_loading.categorical_data.to_csv("./Data/cat_var.csv", encoding='utf-8', index=False)
# Create a corpus from Categorical Data
"""
Corpus is a collection od written or spoken natural language material, 
stored on computer, and used to find out how language is used. 
"""
corpus = pd.read_csv("Data/cat_var.csv")
print(corpus.head(5))
X_cat_var = corpus.drop(['deposit'], axis=1)
y_target = corpus.drop(corpus.iloc[:, 0:9], axis=1)
y_target.columns = ['deposit']

X_cat_var.to_csv("./Data/cat_var2.csv", encoding='utf-8', index=False)
X_cat_var2 = pd.read_csv("Data/cat_var2.csv", sep="\t", header=None)
X_cat_var2.columns = ['body_text']
print(X_cat_var2.head(5))
print(y_target.head(5))

# Put all together
corpus = pd.concat([y_target['deposit'], X_cat_var2['body_text']], axis=1)
print(corpus.head())

# Label Encoding the Target
corpus.columns = ['subscribe', 'cat_var']
corpus['subscribe'] = np.where(corpus['subscribe'] == 'yes', 1, 0)
print(corpus.head(5))



print("---------------------------------------------------------")
print("-- Running Time : %s seconds "% (time.time() - start_time))
print("---------------------------------------------------------")
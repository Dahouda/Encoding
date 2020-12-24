import pandas as pd 
import numpy as np 

print("Data Loading....")

data = pd.read_csv("Bank_Dataset.csv")
print(data.head())
print(data.shape)

# Separate the numeric and categorical variables
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

print("Numeric Variable")
print(numeric_data.head())
print("Shape of Numeric Data :", numeric_data.shape)


print("Categorical Variable")
print(categorical_data.head())
print("Shape of Numeric Data :", categorical_data.shape)
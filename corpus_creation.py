import pandas as pd
import numpy as np
import data_loading


# Import The Data Subset
data_loading.numeric_data.head()
data_loading.categorical_data.head()

# What is the shape of the Categorical Subset
print("Input Data has {} rows and {} columns".format(len(data_loading.categorical_data), len(data_loading.categorical_data.columns)))
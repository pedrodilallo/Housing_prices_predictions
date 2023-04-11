# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

display(train.sample(3))
display(test.sample(3))

# # Sales price distribution

sns.histplot(data=train["SalePrice"]);
train["SalePrice"].describe()

# # Cleaning train dataframe

print("Percentage of empty values by column\n")
print(f"{train.isna().sum() / len(train) * 100}")
print(f"The dataframe has {len(train)} rows")

# The variables Alley, PoolQC, Fence, MiscFeature have more than 80% of empty values, thus they could be droped



# # Separating in two dataframes
#
# to make our analisys easier, we decided to split the data frame in two: one with all the numerical values, to make correlation and principal component analysis and another with all the qualitative or categorical data

train_num_values = train.select_dtypes(["int64","float64"])
train_disc_values = train.select_dtypes(["object"])

# # Quantitative Analysis

# +
# making  a PCA with the numeric values of the dataframe 
display(train_num_values.sample(5))
from sklearn.preprocessing import StandardScaler

# scaling the data for unit variance
train_num_values_no_NA = train_num_values.dropna()
scaler = StandardScaler()
scaler.fit(train_num_values_no_NA)
scaled_data = scaler.transform(train_num_values_no_NA)

#fitting the pca
pca_train = PCA(n_components = 5)
pca_train.fit(scaled_data)

# checking data
x_pca = pca_train.transform(scaled_data)
print(x_pca.shape)


# -
# Creating a heatmap to visualize the pca 
map = pd.DataFrame(pca_train.components_,columns = train_num_values.columns)
plt.figure(figsize = (12,6))
sns.heatmap(map)

# +
train_num_values["YrSold"] = train_num_values["YrSold"].astype("object")
saleprice = train_num_values.groupby("YrSold")["SalePrice"].mean()
plt.plot(saleprice)

# -
# # Correlogram

# Numerical variables only

plt.figure(figsize=(30,20))
sns.heatmap(train_num_values.corr(), annot=True, cmap='coolwarm')
plt.show()

# # Categorical variables

# Categorical columns

train_disc_values.columns

# There are 43 categorical variables

len(train_disc_values.columns)



#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction
# 
# Need to go back through and make adjustments to it

# In[1]:


## import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[18]:


## import the data
df = pd.read_csv("C:\\Users\\boydd\\Downloads\\train (1).csv")


# In[19]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


correlation = df.corr()


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[8]:


# Load the model and fit before doing feature engineering
from sklearn.ensemble import GradientBoostingRegressor


# In[31]:


# drop all null values
# df = df.dropna()


# In[20]:


# There are a lot of columns, i will look at columns that have categorical values
df['MSZoning'].value_counts()

# possible values are RL, RM, FV, RH, C (all)
# Map these values to columns
df = pd.get_dummies(df, columns=["MSZoning"])


# In[21]:


df = df.drop(['Alley','MiscFeature'], axis=1)


# In[22]:


df.info()


# In[23]:


# Fill fence, PoolQC, FireplaceQu Nulls with 0
df['PoolQC'] = df['PoolQC'].fillna(0)
df['Fence'] = df['Fence'].fillna(0)
df['FireplaceQu'] = df['FireplaceQu'].fillna(0)


# In[24]:


df['SalePrice'].describe()


# In[46]:


sns.displot(df['SalePrice'], kind='kde')


# In[25]:


# Can see the SalePrice is skewed 
# May want to log transform it
df.head()


# In[26]:


df.corr()['SalePrice'].sort_values(ascending = False)
# OverallQuall, GrLivArea, GarageCars, GarageArea, TotalBsmtSF are most important


# In[51]:


pd.options.display.max_columns = 82


# In[59]:


df['LotConfig'].value_counts()


# In[27]:


from sklearn.preprocessing import OneHotEncoder


# In[28]:


# This will hold all categorical variables
cat = df.select_dtypes(include='O').keys()
cat


# In[29]:


# plotting all categorical columns vs saleprice

plt.figure(figsize= (30,60))
for i,col in enumerate(cat):
    plt.subplot(13,4,i+1)
    sns.barplot(x=col, y="SalePrice", data=df)
    
    
# In LotShape, IR2 is higher, in landContour, Bnk is lower, 


# In[30]:


# create a function that will be used to one hot encode all of the categorical columns
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# In[31]:


# Get a list of all columns and their type
df = one_hot_encoder(df, cat, drop_first=True)


# In[32]:


# Map all categorical columns at once
df.head()


# In[33]:


df.isnull().sum()
df = df.dropna()


# In[34]:


df.shape


# In[77]:


# separate data into lables, X and y
X = df.drop(['SalePrice'], axis=1)
y = np.log(df['SalePrice'])


# In[36]:


X.isnull().sum()


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[38]:


model = GradientBoostingRegressor()


# In[79]:


model.fit(X_train, y_train)


# In[80]:


training_prediction = model.predict(X_train)


# In[81]:


score_1 = metrics.r2_score(y_train, training_prediction)


# In[82]:


score_2 = metrics.mean_absolute_error(y_train, training_prediction)

print("R2: ", score_1)
print("MAE: ", score_2)


# In[87]:


np.exp(training_prediction)


# In[83]:


test_pred = model.predict(X_test)


# In[84]:


score_3 = metrics.r2_score(y_test, test_pred)
score_4 = metrics.mean_absolute_error(y_test, test_pred)
print(score_3, "\n", score_4)


# In[45]:


### Now see how test holds up
df_test = pd.read_csv("C:\\Users\\boydd\\Downloads\\test (1).csv")


# In[46]:


df_test = df_test.drop(['Alley', 'MiscFeature'], axis=1)


# In[47]:


df_test = pd.get_dummies(df_test, columns=["MSZoning"])


# In[48]:


cat_test = df_test.select_dtypes(include='O').keys()


# In[49]:


df_test = one_hot_encoder(df_test, cat_test, drop_first=True)


# In[ ]:


df_test = df_test.fillna(0)


# In[50]:


missing_cols = set(df.columns) - set(df_test.columns)


# In[51]:


for c in missing_cols:
    df_test[c] = 0

# ensure the order of the test columns is the same in the training set
df_test = df_test[df.columns]


# In[66]:


missing_cols_train = set(df_test.columns) - set(X.columns)
missing_cols_train


# In[69]:


df_test = df_test.drop(['SalePrice'],axis=1)


# In[85]:


model.predict(df_test)


# In[88]:


predictions = model.predict(df_test)


# In[89]:


predictions = np.exp(predictions)


# In[90]:


sub = pd.DataFrame()
sub['Id'] = df_test.Id


# In[91]:


sub['SalePrice'] = predictions
sub.head()


# In[92]:


sub.to_csv("C:\\Users\\boydd\\Downloads\\Housing_Submission1.csv", index=False)


# In[57]:


df_test.head()


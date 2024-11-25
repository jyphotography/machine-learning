# %% [markdown]
# #

# %% [markdown]
# # Problem Statement
# Since I move to New York and look for a house to buy. I want to use this Kaggle dataset to train a model to help me predict the house price.
# https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market

# %% [markdown]
# # 1. EDA

# %%
import pandas as pd
import numpy as np

import seaborn as sns
# from matplotlib import pyplot as plt

# %% [markdown]
# ## 1.1 Data Loading

# %%
df = pd.read_csv('../data/NY-House-Dataset.csv')

# %%
# df.head()

# %%
# df.info()

# %% [markdown]
# ## 1.2 Data Cleaning

# %%
df.duplicated().sum()

# %%
df.drop_duplicates(inplace=True)

# %%
df.duplicated().sum()

# %% [markdown]
# ### Check Price Distribution

# %%
# plt.figure(figsize=(6, 4))

# sns.histplot(df['PRICE'], bins=40, color='black', alpha=1)
# plt.ylabel('Frequency')
# plt.xlabel('Price')
# plt.title('Distribution of prices')

# plt.show()

# %% [markdown]
# ### Remove Extreme Values

# %%
# use interquartile range to remove price outliers
upper_limit = df['PRICE'].quantile(0.94) # 6MM
lower_limit = df['PRICE'].quantile(0.02) # 170K
baths_limit = df['BATH'].quantile(0.99) # 8 BATHS
beds_limit = 10

# %%
outliers = df[(df['PRICE'] < lower_limit) | (df['PRICE'] > upper_limit) | (df['BATH'] > baths_limit) | (df['BEDS'] > beds_limit)]
# drop rows containing outliers
df_new = df.drop(outliers.index)

# %%
# plt.figure(figsize=(6, 4))

# sns.histplot(df_new['PRICE'], bins=40, color='black', alpha=1)
# plt.ylabel('Frequency')
# plt.xlabel('Price')
# plt.title('Distribution of prices')

plt.show()

# %%
df_new.describe()

# %%
df_new['BEDS'].value_counts()

# %%
df_new.SUBLOCALITY.value_counts()

# %%
# Remove rows where the values in 'col1' appear less than 10 times
df_new = df_new.groupby('SUBLOCALITY').filter(lambda x: len(x) >= 10)

# %%
# df_new.SUBLOCALITY.value_counts()

# %%
# Calculate skewness for each column
# for column in df.select_dtypes(include=[np.number]).columns:
#     skewness_per_column = df_new[column].skew()
#     print(f"Skewness for {column}: {skewness_per_column}")

# %% [markdown]
# ## 1.3 Check Features Correlation

# %%
# correlationn=pd.DataFrame()
# for column in df_new.select_dtypes(include=[np.number]).columns:
#     correlationn[column] = df_new[column]
# relations=correlationn.corr()
# sns.heatmap(relations, annot=True, cmap='coolwarm')
# plt.show()


# %% [markdown]
# I will use the BED, BATH, PROPERTYSQFT these 3 features due to high correlationn

# %% [markdown]
# # 2. Model Training

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

# %%
selected_cols = ['BEDS', 'BATH', 'PROPERTYSQFT','SUBLOCALITY','PRICE']
subset_df = df_new[selected_cols]

# %%
subset_df.head()

# %%
def train_val_test_split(df, target, train_size, val_size, test_size, random_state):
    
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    val_portion = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(df_full_train, test_size=val_portion, random_state=random_state)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train[target].values
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    del df_full_train[target]
    del df_train[target]
    del df_val[target]
    del df_test[target]

    return df_full_train, df_train, df_test, df_val, y_full_train, y_train, y_val, y_test

# %%
X_full_train, X_train, X_test, X_val, y_full_train, y_train, y_val, y_test = \
    train_val_test_split(df=subset_df, target='PRICE', train_size=0.8, val_size=0.1, test_size=0.1, random_state=1)

# %%
def train_rf(df_train, y_train, random_state, n_estimators):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

# %%
dv, rf_model = train_rf(df_train=X_train, y_train=y_train, random_state=1, n_estimators=10)
y_pred = predict(df=X_test, dv=dv, model=rf_model)
round(rmse(y_test, y_pred),3)

# %%
y_pred,y_val
compare = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
pd.set_option('display.float_format', '{:.0f}'.format)
# Display the DataFrame
print(compare)

# %% [markdown]
# ## 2.1 Performance Tuning

# %%
def train_rf2(df_train, y_train, n_estimators, depth):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts)

    model = RandomForestRegressor(max_depth=depth, n_estimators=n_estimators, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model

# %%
all_rmse = {}
for depth in [10, 15, 20, 25]:
    print('depth: %s' % depth)
    rmse_vals = []
    for i in range(10, 201, 10):
        dv, rf_model = train_rf2(df_train=X_train, y_train=y_train, n_estimators=i, depth=depth)
        y_pred = predict(df=X_val, dv=dv, model=rf_model)
        rmse_val = round(rmse(y_val, y_pred),3)
        # print('%s -> %.3f' % (i, rmse_val))
        rmse_vals.append(rmse_val)
    all_rmse[depth] = np.mean(rmse_vals)

# %%
all_rmse

# %% [markdown]
# depth use 20 

# %%
all_rmse_dep20 = {}

for i in range(10, 201, 10):
    dv, rf_model = train_rf2(df_train=X_train, y_train=y_train, n_estimators=i, depth=10)
    y_pred = predict(df=X_val, dv=dv, model=rf_model)
    rmse_val = round(rmse(y_val, y_pred),3)
    print('%s -> %.3f' % (i, rmse_val))
    all_rmse_dep20[i] = rmse_val

# %% [markdown]
# n_estimator use 10

# %% [markdown]
# ## 2.2 Feature Importance

# %%
dv, rf_model = train_rf2(df_train=X_train, y_train=y_train, n_estimators=10, depth=10)

# %%
# Get feature importances
importances = rf_model.feature_importances_

# Print feature importances
feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# %% [markdown]
# ## 3. Export the model

# %%
import pickle

# %%
# Save the model to a file
with open('model.bin', 'wb') as f:
    pickle.dump(rf_model, f)
with open('dv.bin', 'wb') as f:
    pickle.dump(dv, f)

# %%
# To load the model
with open('model.bin', 'rb') as f:
    loaded_model = pickle.load(f)
with open('dv.bin', 'rb')as f_in:
    loaded_dv = pickle.load(f_in)

# %%
X_test.head()

# %%
# Predict Price
test_json = {"BEDS": 4, "BATH": 2, "PROPERTYSQFT": 2184, "SUBLOCALITY": "Queens County"}
X = loaded_dv.transform([test_json])
loaded_model.predict(X)

# %%
# Actual Price
y_test[0]



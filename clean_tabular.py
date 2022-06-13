import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#Load the dataset from CSV
pd.set_option('display.max_columns', None)
products_df = pd.read_csv("Products.csv", lineterminator='\n')

#Visualize data
products_df.head()
products_df.describe()
products_df.info()
products_df.columns
#For a categorical dataset we want to see how many instances of each category there are #Use this later!
#products_df['category'].value_counts()

#Split Catagory Column to most general Category that the product exits in, then map to unique Int in new df column
result = [x.split(" / ")[0] for x in products_df['category']]
products_df['category_unique'] = result
mapping = {item:i for i, item in enumerate(products_df["category_unique"].unique())}
products_df["category_unique"] = products_df["category_unique"].apply(lambda x: mapping[x])

#Correct Price Column to remove leading £ sign, remove comma, and change data type to float then convert to int
products_df['price'] = products_df['price'].str.strip('£')
products_df['price'] = products_df['price'].str.replace(',', '')
products_df['price'] = products_df['price'].astype('float64')
products_df['price'] = products_df['price'].astype('int')

#Remove all non AlphaNumeric Characters from both the product name and product description!
products_df['product_name'] = products_df['product_name'].str.replace('\r', ' ', regex=False)
products_df['product_description'] = products_df['product_description'].str.replace('\r', ' ', regex=False)
products_df['product_name'] = products_df['product_name'].str.replace('\W\s', '', regex=True)
products_df['product_description'] = products_df['product_description'].str.replace('\W\s', '', regex=True)

#Prints All unique values of the data columns we are interested in
print(np.sort(products_df['price'].unique()))
print(np.sort(products_df['product_name'].unique()))
print(np.sort(products_df['product_description'].unique()))
print(np.sort(products_df['location'].unique()))
print(np.sort(products_df['category'].unique()))

cvec = CountVectorizer()

x = products_df['product_name']
y = products_df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
cvec = CountVectorizer(stop_words='english').fit(x_train)
name_train = pd.DataFrame(cvec.transform(x_train).todense(), columns=cvec.get_feature_names_out())
name_test = pd.DataFrame(cvec.transform(x_test).todense(), columns=cvec.get_feature_names_out())

x = products_df['product_description']
y = products_df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
cvec = CountVectorizer(stop_words='english').fit(x_train)
description_train = pd.DataFrame(cvec.transform(x_train).todense(), columns=cvec.get_feature_names_out())
description_test = pd.DataFrame(cvec.transform(x_test).todense(), columns=cvec.get_feature_names_out())

x = products_df['location']
y = products_df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
cvec = CountVectorizer(stop_words='english').fit(x_train)
location_train = pd.DataFrame(cvec.transform(x_train).todense(), columns=cvec.get_feature_names_out())
location_test = pd.DataFrame(cvec.transform(x_test).todense(), columns=cvec.get_feature_names_out())

#train = pd.concat([name_train, description_train, location_train], axis=1)
#test = pd.concat([name_test, description_test, location_test], axis=1)
#lr = LogisticRegression(max_iter=10000)
#lr.fit(train, y_train)
#print(lr.score(test, y_test))

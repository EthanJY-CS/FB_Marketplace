import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

pd.set_option('display.max_columns', None)
products_df = pd.read_csv("Products.csv", lineterminator='\n')
#Correct Price Column to remove leading £ sign, remove comma, and change data type to float
products_df['price'] = products_df['price'].str.strip('£')
products_df['price'] = products_df['price'].str.replace(',', '')
products_df['price'] = products_df['price'].astype('float64')
products_df['price'] = products_df['price'].astype('int')

#Remove all non AlphaNumeric Characters from both the product name and product description!
products_df['product_name'] = products_df['product_name'].str.replace('\W\s', '', regex=True)
products_df['product_description'] = products_df['product_description'].str.replace('\W\s', '', regex=True)

#Prints All unique values of the data columns we are interested in
#print(np.sort(products_df['price'].unique()))
#print(np.sort(products_df['product_name'].unique()))
#print(np.sort(products_df['product_description'].unique()))
#print(np.sort(products_df['location'].unique()))

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

train = pd.concat([name_train, description_train, location_train], axis=1)
test = pd.concat([name_test, description_test, location_test], axis=1)
lr = LogisticRegression(max_iter=10000)
lr.fit(train, y_train)
print(lr.score(test, y_test))

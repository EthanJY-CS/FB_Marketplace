import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

#Load the dataset from CSV
pd.set_option('display.max_columns', None)
products_df = pd.read_csv("Products.csv", lineterminator='\n')
images_df = pd.read_csv("Images.csv", lineterminator='\n')

def read_DataFrame(df):
    print(df.head())
    df.describe()
    df.info()
    df.columns

#read_DataFrame(products_df)
#read_DataFrame(images_df)

def get_unique(df):
    for column in df.columns:
        print(column)
        print(np.sort(df[column].unique()))

#get_unique(products_df)
#get_unique(images_df)

def clean_MergedDataframe():
    merged_df = products_df.merge(images_df[['product_id']], left_on='id',
                  right_on='product_id').drop(['id', 'url', 'page_id', 'create_time', 'product_id'], axis=1)
    merged_df = merged_df.iloc[: , 1:]
    merged_df = merged_df.astype('string')
    merged_df['price'] = merged_df['price'].str.replace('[£,]', '', regex=True)
    merged_df['price'] = merged_df['price'].astype('float64')
    merged_df['price'] = merged_df['price'].astype('int')
    merged_df['product_name'] = merged_df['product_name'].str.lower().replace('\W', ' ', regex=True)
    merged_df['product_name'] = merged_df['product_name'].str.replace('\s+', ' ', regex=True)
    merged_df['product_description'] = merged_df['product_description'].str.lower().replace('\W', ' ', regex=True)
    merged_df['product_description'] = merged_df['product_description'].str.replace('\s+', ' ', regex=True)
    merged_df['location'] = merged_df['location'].str.lower().replace(',', '', regex=True)
    result = [x.split(" / ")[0] for x in merged_df['category']]
    merged_df['category'] = result
    merged_df['category'] = merged_df['category'].astype('category').cat.codes
    return merged_df

merged_df = clean_MergedDataframe()
#read_DataFrame(merged_df)
#get_unique(merged_df)

def text_Regression_model():
    y = merged_df['price']
    X = merged_df['product_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cvec = CountVectorizer(stop_words='english').fit(X_train)
    name_train = pd.DataFrame(cvec.transform(X_train).todense(), columns=cvec.get_feature_names_out())
    name_test = pd.DataFrame(cvec.transform(X_test).todense(), columns=cvec.get_feature_names_out())
    X = merged_df['product_description']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cvec = CountVectorizer(stop_words='english').fit(X_train)
    description_train = pd.DataFrame(cvec.transform(X_train).todense(), columns=cvec.get_feature_names_out())
    description_test = pd.DataFrame(cvec.transform(X_test).todense(), columns=cvec.get_feature_names_out())
    X = merged_df['location']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cvec = CountVectorizer(stop_words='english').fit(X_train)
    location_train = pd.DataFrame(cvec.transform(X_train).todense(), columns=cvec.get_feature_names_out())
    location_test = pd.DataFrame(cvec.transform(X_test).todense(), columns=cvec.get_feature_names_out())

    train = pd.concat([name_train, description_train, location_train], axis=1)
    test = pd.concat([name_test, description_test, location_test], axis=1)
    train = train.loc[:,~train.columns.duplicated()].copy()
    test = test.loc[:,~test.columns.duplicated()].copy()

    lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)
    lr.fit(train, y_train)
    print(f'The score was: {lr.score(test, y_test)}')

#text_Regression_model()

def image_Classification_model():
    image_data = np.load('images.npy')
    X = image_data[:-1]
    print(X.shape)
    y = merged_df['category'].to_numpy()
    print(y.shape)
    X_flat = np.array(X).reshape((12604, 256*256*3))
    print(X_flat.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    # Provide chunks one by one
    chunkstartmarker = 0
    chunksize = 500
    numtrainingpoints = len(X_train)
    model = SGDClassifier(loss='log_loss')
    while chunkstartmarker < numtrainingpoints:
        X_chunk = X_train[chunkstartmarker:chunkstartmarker+chunksize]
        y_chunk = y_train[chunkstartmarker:chunkstartmarker+chunksize]
        model.partial_fit(X_chunk, y_chunk, np.unique(y))
        chunkstartmarker += chunksize
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

image_Classification_model()


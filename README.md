# FaceBook Marketplace

> The Facebook Marketplace Recommendation Architecture

## Milestone 1: An Overview Of The System

The link to the video of the overview can be found at:
[Click Here for Overview](https://www.youtube.com/watch?v=1Z5V2VrHTTA&t=2s)

To create the same architecture that FaceBook Marketplace uses to rank search result. Combining pre-trained neural networks that process images and text into a single
multi-modal machine learning model, and build the cloud infrastructure around it to serve predictions to a search index through an API.

## Milestone 2: Explore The Dataset

Prerequisit content:

Products.csv
- Tabular Data of the products including; id, product_name, category, product description, price, location, url, page_id, create_time

Images.csv
- Tabular Data of the Images including, id, product_id, bucket_link, image_ref, create_time

All Images are downloaded from an S3 bucket, as in Images.csv, the bucket_link is provided to each Image. A Total of 12664 Images are then stored in a
directory named Images where each image named after it's id.

Objectives:

> Clean Tabular Data ---> clean_tabular.py

First thing we do is to perform Exploratory Data Analysis (EDA) on the product Data. So we first load both Images.csv (images_df) and Products.csv (products_df)
into their own respective pandas Dataframes. This then allows us to then explore the data using pandas using typical methods of viewing the data such as df.head(),
df.describe(), df.info(), df.columns, but also using numpy, where we can use .unique() to give us the unique values in each data column.

Performing EDA is necessary because data is very messy, it is very unlikely that you have data that is clean and ready to be served into models straight away
so we must explore and find ways to clean and preprocess our data before moving on. So one of the important factors to look for first is missing data, this can be
NaN values, missing data, or any other symbol used in place of where relevent data would be. Using the df.info() and numpy .unique() we can see how many non values
exist or if a symbol is used in place, and in this case, there were no missing data fields, but there could have been!...

The fields that we will be concerned with are the product_name, product_description, location, and price (All used for text models and embeddings later) but first,
we need to merge the 2 dataframes of images and products as there are 12604 images that all correspond to 7156 products. So using the pandas merge method, we
merge the 2 dataframes matching on products_df id field, and images_df product_id, with sorting = True (Sorts in alphabetical on the merged id column) then dropping
all columns leaving a merged_df that contains product_name, category, product_description, price and location, except this is now of length 12604, each entry an image
and their products data (so yes there is now repeating data for products that have multiple images, but this is cleaned mostly as we have dropped all the fields we don't need)

Now we have just the fields we are wanting, it's time to clean the data that is left. This includes proper data typing, removing/cleaning text data, and transforming
categorical data. I first converted every column of data to string datatype as a starting point.
- merged_df['price'] - For price, we stripped the leading £ sign and the comma from each entry and converted to float then to int as it converted nicely that way
- merged_df['product_name'] - For name, we stripped leaving only alphanumeric characters and converted whitespaces to a single whitespace using regex and made it lowercase
- merged_df['product_description'] - For description, performed the same above as to product name.
- merged_df['location'] - For location, made to lowercase
- merged_df['category'] - For category, it was a little more involved, as the categories are named by all it's abrastrated categories a product lies in. For example
Music, Films, Books & Games / Music / CDs where / denotes each sub category that follows. In our instance, we take the most broad category which is the first section. So we use .split('/')[0]
to then apply that to each df entry and then we convert this categorical data to a unique category code using pandas .cat.codes. Which then gives us a total of 13 unique categories that products belong to

This gives us a merged_df with clean data and text fields that have been preprocessed for Natural Language Processing (NLP) however, the text fields are still text,
therefore what will follow in the upcoming task is to transform the merged dataframe text fields into numerical data ready for simple machine learning models.

> Clean Image Data ---> clean_images.py

First, I set out to find out if every image found in the Images directory had a corresponding product, if it didn't then it's not needed and therefore
deleted. I achieved this by loading the Images.csv into a pandas Dataframe, and each image is named after the id and therefore product it belongs to. So, I iterated
through all the images, finding if the image existed in the image_df['id'], taking advantage of the .unique() method as products could have multiple images.
This found that 60 images in total didn't have a product matched to, and therefore were deleted from the Images directory.

We now have 12604 images of all varying sizes, so we need to clean the image data in a way to resize all images to a uniformed size. After finding out
the minimum height and width of an image by creating a quick method that goes through all images, resizing to a uniformed 256x256 was okay. This was
achieved by resizing each image by the ratio of its former size, pasted on a blank black square if the image were smaller in dimensions along
width or height.

I then created a method that saved these images into a numpy Array, this was for the upcoming models in the future, but it made sense to save an array of the image
data now rather than always creating the image data upon running every time. images.npy, contains the image data of shape (12604, 256, 256, 3) --->
(# of Images, Width, Height, Colour Channels)

## Milestone 3: Create Simple Machine Learning Models

```python
    """
        Code Example Template
    """
```

## Milestone 4: Create The Vision Model

## Milestone 5: Create The Text Understanding Model

## Milestone 6: Combine The Models

## Milestone 7: Setup KubeFlow

## Milestone 8: Deploy The Model To KubeFlow

## Final comments

from PIL import Image
import glob
import numpy as np
import pandas as pd
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def remove_Unmatched_Images_ByID():
    filelist = glob.glob('images/*.jpg')
    id_list = []
    for file in filelist:
        str = file.replace('images/', '')
        str = str.replace('.jpg', '')
        id_list.append(str)
    images_df = pd.read_csv("Images.csv", lineterminator='\n')
    for id in id_list:
        if id not in images_df['id'].unique():
            os.remove('images/' + id + '.jpg')

def save_Images_toNumpy():
    filelist = glob.glob('cleaned_images/*.jpg')
    temp = np.array([np.array(Image.open(fname)) for fname in filelist])
    np.save('images', temp)

def min_Dimensions_of_Image():
    path = "images/"
    dirs = os.listdir(path)
    min_Height = 999999
    min_Width = 999999
    for n, item in enumerate(dirs, 1):
        im = Image.open('images/' + item)
        h = im.size[0]
        w = im.size[1]
        if h < min_Height:
            min_Height = h
        if w < min_Width:
            min_Width = w

    print(f'The minimum Height was: {min_Height}')
    print(f'The minimum Width was: {min_Width}')

if __name__ == '__main__':
    remove_Unmatched_Images_ByID()
    #Find min Dimensions and then choose a valid FinalSize
    #min_Dimensions_of_Image()
    path = "images/"
    dirs = os.listdir(path)
    dirs = sorted(dirs)
    #Choosing 256 as FinalSize after evaluating the Min dimensions
    final_size = 256
    for n, item in enumerate(dirs, 1):
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{n}_resized.jpg')
    save_Images_toNumpy()

    

    

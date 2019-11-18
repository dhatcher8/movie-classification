import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
from os import path

ImageFile.LOAD_TRUNCATED_IMAGES = True

#if(~path.exists('movies_metadata_processed.csv')):
input_file = pd.read_csv('movies_metadata.csv')
num_rows = len(input_file.index)
# Creating a new dataframe with only the columns that we want
outdf = pd.DataFrame(index=range(num_rows), columns=['id', 'title', 'genres', 'posterpath'])
outdf['id'] = input_file['id']
outdf['title'] = input_file['title']
outdf['genres'] = input_file['genres']
outdf['posterpath'] = input_file['poster_path']
# outdf = outdf.set_index('id')

nullGenres = outdf[outdf['genres'] == '[]'].index
outdf.drop(nullGenres, inplace=True)
nullPosters = outdf[outdf['posterpath'].isnull()].index
outdf.drop(nullPosters, inplace=True)
dropIds = ['106605','121351','140470','156415','1997-08-20','2012-09-29','2014-01-01','215908','23022','31772','35810','38585','53571','55602','56325','77621','79968']
outdf = outdf[~outdf['id'].isin(dropIds)]

img_width = 185
img_height = 185

# Method to create new dataframe with the an exact amount of movies (cannot exceed around 42000)
movie_count = len(outdf)
selected_movies = outdf#outdf.sample(movie_count)
# selected_movies.columns
selected_movies = selected_movies.reset_index(drop=True)
# # selected_movies	

X = []
# outdf['id'][1]
# print(selected_movies.iloc[1, 0])

if(~path.exists('img_arr.npy')):
	for i in tqdm(range(selected_movies.shape[0])):
	  # try:
	  #path = '/content/drive/My Drive/Movie Poster ML/data/PosterImages/' + selected_movies.iloc[i, 0] + '.jpg'
	  path = 'PosterImages/PosterImages/' + selected_movies.iloc[i, 0] + '.jpg'
	  img = image.load_img(path, target_size=(img_width, img_height, 3))
	  img = image.img_to_array(img)
	  img = img/255.0
	  X.append(img)
	  # except:
	    # print(selected_movies.iloc[i, 0])

	print("Converting to Numpy Array...")
	X = np.array(X)
	print("NP Array created, saving...")
	#plt.imshow(X[1])
	#plt.show()

	np.save('img_arr', X)
	print("Saved to img_arr.npy")
else:
	print("Loading images...")
	X = np.load('img_arr.npy')
	print(X.shape)
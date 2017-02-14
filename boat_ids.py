import prepare_files
from scipy.misc import imread
import numpy as np
import pandas as pd
import show_images
import matplotlib.pyplot as plt
from sklearn import cluster
import cv2
import multiprocessing

# just get the files:
# image1=prepare_files.train_array[:,0][1]
# print(image1)



# one=np.array(imread(image1))
# train = np.array([imread(img) for img in train_files])
# print(one.shape)


all_files=prepare_files.train_array[:,0]
# let's take random subset to speed up process, as per anokas
sampled_files=np.random.choice(all_files,50,replace=False)

# come back and just run on all_files when done testing. will take time
train = np.array([imread(img) for img in sampled_files])

# what sizes do we have?
print('Sizes in train:')
shapes = np.array([str(img.shape) for img in train])
pd.Series(shapes).value_counts()


print(train[1].shape)
# SHOW IMAGES of certain sizes
# for uniq in pd.Series(shapes).unique():
#     show_images.show_four(train[shapes == uniq], 'Images with shape: {}'.format(uniq))
#     plt.show()


# # CLUSTERING

# # Function for computing distance between images
# def compare(args):
#     img, img2 = args
#     img = (img - img.mean()) / img.std()
#     img2 = (img2 - img2.mean()) / img2.std()
#     return np.mean(np.abs(img - img2))

# # Resize the images to speed it up.
# train = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in train]

# # Create the distance matrix in a multithreaded fashion
# pool = multiprocessing.Pool(8)
# #bar = progressbar.ProgressBar(max=len(train))
# distances = np.zeros((len(train), len(train)))
# for i, img in enumerate(train): #enumerate(bar(train)):
#     all_imgs = [(img, f) for f in train]
#     dists = pool.map(compare, all_imgs)
#     distances[i, :] = dists


# cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)
# y = cls.fit_predict(distances)
# print(y)
# print('Cluster sizes:')
# print(pd.Series(y).value_counts())

# # tack y onto training dataframe
# boat_id_df=np.c_[sampled_files,y]

# # convert to pd if it helps with merges
# # pddf=pd.DataFrame(boat_id_df)

# # when i run with all images, i will want to concatenate 
# # image path and class for every image. this will feed into another
# # file which will train classifier model

# print(train_array.shape)


# # MORE image plots
# # for uniq in pd.Series(y).value_counts().index:
# #     if uniq != -1:
# #         size = len(np.array(train)[y == uniq])
# #         if size > 5:
# #             show_images.show_eight(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))
# #             plt.show()
# #         else:
# #             show_images.show_four(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))
# #             plt.show() 
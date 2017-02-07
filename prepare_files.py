import pandas as pd
import numpy as np
import os 


# training images
images_dir = '/Users/wnowak/gitt/kaggle_nature_conserv/train/'

species=[images_dir+f for f in os.listdir(images_dir)[1:]]


file_paths=[]
fish_type=[]
for i in range(len(species)):
    for j in range(len(os.listdir(species[i]))):
        images=os.listdir(species[i])
        file_paths.append(species[i]+'/'+images[j])
        fish_type.append(species[i][-3:])
#         file_path.append(species[i]+os.listdir(species[j])

# print(file_paths[1:10])

file_array=np.asarray(file_paths)
fish_array=np.asarray(fish_type)

# combine into array for training
train_array=np.c_[file_array,fish_array]

# just get the files:

# train_array[:,0]
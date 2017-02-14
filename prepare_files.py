import pandas as pd
import numpy as np
import os 


# training images
images_dir = '/Users/wnowak/gitt/kaggle_nature_conserv/train/'

species=[images_dir+f for f in os.listdir(images_dir)[1:]]

all_images=[]
all_fishes=[]
for i in range(len(species)):
	images=[]
	fish_type=[]
	for j in range(len(os.listdir(species[i]))):
		images.append(os.listdir(species[i])[j])
		fish_type.append(species[i][-3:])
	all_images.append(images)
	all_fishes.append(fish_type)


	# for j in range(len(os.listdir(species[i]))):
 #    		file_paths.append(species[i]+'/'+images[j])
 #    		fish_type.append(species[i][-3:])
 #    		if j%1000==0:
 #    			print(j)
 #    			print(fish_type[j])
 #    			print(file_paths[j])

# print(len(all_images[7]))
# print(all_fishes)


image_array=np.asarray(sum(all_images,[]))
fish_array=np.asarray(sum(all_fishes,[]))

d = {'image': image_array, 'fish type': fish_array}
df = pd.DataFrame(data=d)
df['full path']='/Users/wnowak/gitt/kaggle_nature_conserv/train/'+df['fish type']+'/'+df['image']


# print(df.shape)
# print(df.head)

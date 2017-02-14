 # tf nn

import prepare_files
from scipy.misc import imread
import numpy as np
import pandas as pd
import show_images
import matplotlib.pyplot as plt
from sklearn import cluster
import cv2
import multiprocessing
import tensorflow as tf

# get all file paths in pd array
df=pd.DataFrame(prepare_files.train_array)
df.columns = ['file path', 'word label']
# take random sample so this is more managable for now
# come back and just run on all_files when done testing. will take time
sampled_df=df.sample(50)

# read the file paths into list called imgs
imgs=[imread(img) for img in sampled_df['file path']]

#resize images to all 1100 by 500 pxls
width=28
height=28
dim=(width,height)
resized_images = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in imgs]



# convert to np array
train = np.array(resized_images) #.shape = (10,500,1100,3)
train=train.astype(np.float32)
train_labels = pd.get_dummies(sampled_df['word label'])
# convert to np for tf
train_labels_values=train_labels.values



# sess=tf.Session()

image_size1=width
image_size2=height
input_placeholder = tf.placeholder(tf.float32, shape=[None, image_size1, image_size2, 3])


patch_size = 5
depth = 16
num_channels = 3 # grayscale
num_labels=10

# layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1),name="layer1_weights")
# tf.global_variables_initializer()
# conv = tf.nn.conv2d(train, layer1_weights, [1, 2, 2, 1], padding='SAME')

# relu_out=tf.nn.relu(conv,name='firstlayer')
# result=sess.run(relu_out)






# follow deep MNIST experts tutorial

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

# x_image = tf.reshape(x, [-1,28,28,1])


h_conv1 = tf.nn.relu(conv2d(train, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

shp=int(width/4 * height/4 * 64)
W_fc1 = weight_variable([shp, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, shp])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, shape=[None, 8])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())



epochs_completed = 0
index_in_epoch = 0
num_examples = train.shape[0]



# serve data by batches
def next_batch(batch_size):
    
    # global train_images
    # global train_labels
    # global index_in_epoch
    # global epochs_completed
    
    # start = index_in_epoch
    # index_in_epoch += batch_size
    
    # # when all trainig data have been already used, it is reorder randomly    
    # if index_in_epoch > num_examples:
    #     # finished epoch
    #     epochs_completed += 1
    #     # shuffle the data
    #     perm = np.arange(num_examples)
    #     np.random.shuffle(perm)
    #     train_images = train[perm]
    #     train_labels = train_labels_values[perm]
    #     # start next epoch
    #     start = 0
    #     index_in_epoch = batch_size
    #     assert batch_size <= num_examples
    # end = index_in_epoch
    return train[1:batch_size], train_labels_values[1:batch_size]


TRAINING_ITERATIONS=1
BATCH_SIZE=3
DROPOUT=.5
for i in range(TRAINING_ITERATIONS):
    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})






























# for i in range(100):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
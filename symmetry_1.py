#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
import ast, math
import time

#cls = np.load('training_data_g7_2_12_6_4_cls.npy')
#data = np.load('training_data_g7_2_12_6_4_data.npy')

cls = np.load('training_data_g5_cls.npy')
data = np.load('training_data_g5_data.npy')

x_train = data
y_train = cls

#print(set(cls))
print(x_train.shape)
print(y_train.shape)


# In[10]:


# Training

start = time.time()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=data[0].shape),
  tf.keras.layers.Dense(1200, activation='relu'),
  tf.keras.layers.Dense(800, activation='relu'),
  #tf.keras.layers.Dense(400, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(set(cls)), activation='softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5)
model.summary()

stop = time.time()

print ("Training completed. Execution time: {}".format(stop - start))


# In[11]:


#cls, data = read_data('test_data.txt')

#cls = np.load('test_data_g7_12_6_4_cls.npy')
#data = np.load('test_data_g7_12_6_4_data.npy')

cls = np.load('test_data_g5_cls.npy')
data = np.load('test_data_g5_data.npy')

print(cls.shape)
print(data.shape)


# In[12]:


# Testing

start = time.time()

i = 0
success = 0
failure = 0

wrong_preds = {key: 0 for key in set(cls)}

for d in data:
    #y_pred = model.predict(d)
    y_pred_class = model.predict_classes(d)
    
    if y_pred_class[0] == cls[i]:
        success = success + 1
    else:
        failure = failure + 1
        print ("class: {}, y_pred: {}".format(cls[i], y_pred_class[0])) #, y_pred[0,:]))
        #print ("class: {}, y_pred: {}, weights: {}".format(cls[i], y_pred_class[0], y_pred[0,:]))
        wrong_preds[y_pred_class[0]] = wrong_preds[y_pred_class[0]] + 1
        wrong_preds[cls[i]] = wrong_preds[cls[i]] + 1
        
    i = i + 1
    
print ("successfull predictions: {}/{}".format(success, len(cls)))
print ("Failure map: {}".format(wrong_preds))
sp = (success/len(cls)) * 100
print ("success percentage: {}%".format(sp))

stop = time.time()

print ("Testing completed. Execution time: {}".format(stop - start))






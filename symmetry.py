#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
import numpy as np
import ast, math

def read_data(filename):
    with open(filename, 'r') as fh:
        cls = []
        data = None
        for ln in fh:
            
            d = ln.split(':')
            d1 = ast.literal_eval(d[0])
            cls.append(d1)
            
            d2 = np.array([ast.literal_eval(d[1].rstrip())])
            
            if data is None:
                data = d2
            else:
                data = np.append(data, d2, axis = 0)
            
        return cls, data

cls, data = read_data('training_data_g5.txt')


# In[19]:


x_train = data
y_train = np.array(cls)

print(set(cls))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=data[0].shape),
  tf.keras.layers.Dense(1200, activation='relu'),
  tf.keras.layers.Dense(800, activation='relu'),
  #tf.keras.layers.Dense(400, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(set(cls)), activation='softmax')
])


# In[20]:


model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)


# In[21]:


history = model.fit(x_train, y_train, epochs=5)


# In[23]:


model.summary()


# In[25]:


t0 = np.array([[[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]])
t1 = np.array([[[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]])
t2 = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]])
t5 = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])
t4 = np.array([[[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
t3 = np.array([[[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]])
t5_1 = np.array([[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
#x_test = t4
#y_pred = model.predict(x_test)
#y_pred_class = model.predict_classes(x_test)

#print (cls[0])

cls, data = read_data('test_data_g5.txt')

i = 0
success = 0
failure = 0

for d in data:
    dn = np.array([d])
    y_pred = model.predict(dn)
    y_pred_class = model.predict_classes(dn)
    
    if y_pred_class[0] == cls[i]:
        success = success + 1
    else:
        failure = failure + 1
        print ("class: {}, y_pred: {}, weights: {}".format(cls[i], y_pred_class[0], y_pred[0,:]))
        
    i = i + 1
    
print ("successfull predictions: {}/{}".format(success, len(cls)))
sp = (success/len(cls)) * 100
print ("success percentage: {}%".format(sp))
#''' 




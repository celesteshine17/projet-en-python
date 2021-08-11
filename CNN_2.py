#!/usr/bin/env python
# coding: utf-8

# In[17]:


#importation des modules
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# In[10]:


#Initialiser la CNN 

classifier = Sequential()


# In[11]:


#Etape 1: Convolution
#Premiere couche de convolution 
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[12]:


# Deuxieme couche de convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[13]:


#Troisieme couche de convolution
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[14]:


#Quatrieme couche de convolution
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[15]:


#Etape 3 - Flattening
classifier.add(Flatten())


# In[18]:


# Etape 4 - couche completement connectée
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[19]:


# Compiler the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


# Partie 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset\\training_set',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset\\test_set',
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 75,
                         validation_data = test_set,
                         validation_steps = 63)


# In[ ]:


# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = "chien"
    print(prediction)
else:
    prediction = "chat"
    print(prediction)
    


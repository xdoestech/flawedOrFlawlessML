import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
import joblib
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
'''
https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
KERAS CNN DOES NEED TO FLATTEN IMAGES
NO HOG
Epoch 10/10 sample size 592 test size 0.1
5/5 [==============================] - 11s 2s/step - loss: 0.6775 - accuracy: 0.5966 - val_loss: 0.6865 - val_accuracy: 0.5500
Epoch 10/10 BATCH SIZE 32
17/17 [==============================] - 11s 662ms/step - loss: 0.6930 - accuracy: 0.5047 - val_loss: 0.6927 - val_accuracy: 0.5000
Epoch 10/10 BATCHSIZE 32
17/17 [==============================] - 13s 763ms/step - loss: 0.5143 - accuracy: 0.7505 - val_loss: 0.5990 - val_accuracy: 0.6667
Accuracy: 66.67%
Epoch 10/10 batch size 32
17/17 [==============================] - 12s 721ms/step - loss: 0.6164 - accuracy: 0.6923 - val_loss: 0.6456 - val_accuracy: 0.6000
Accuracy: 60.00%
Epoch 15/15 batchsize 31 epochs 15
17/17 [==============================] - 13s 746ms/step - loss: 0.6283 - accuracy: 0.6604 - val_loss: 0.6823 - val_accuracy: 0.4833
Accuracy: 48.33%
Epoch 10/10 HIDDEN LAYER 250 125
17/17 [==============================] - 11s 619ms/step - loss: 0.5543 - accuracy: 0.7036 - val_loss: 0.6516 - val_accuracy: 0.6167
Accuracy: 61.67%
'''

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})women images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    if(im.shape == (80,80,3)):
                        data['label'].append(subdir[:])
                        data['filename'].append(file)
                        data['data'].append(im)
                    
 
        joblib.dump(data, pklname)
data_path = fr'{os.getenv("HOME")}/../10sN1s/'
print(os.listdir(data_path))

base_name = 'flawedNflawless'
width = 80
 
include = {'10s', '1s'}

'''RESIZE DATA PRINT STATS ABOUT DATA '''
resize_all(src=data_path, pklname=base_name, width=width, include=include)

from collections import Counter
 
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

Counter(data['label'])

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])
'''data preview below'''
X = np.array(data['data'])
y = np.array(data['label'])
'''PRE PROCESS AND SPLIT DATA TO BE USED'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.1, 
    shuffle=True,
    random_state=42
    # stratify = stratKfold
)
'''KERAS CNN'''
# keras imports for the dataset and building our neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# # building the input vector from the 32x32 pixels
# X_train = X_train.reshape(X_train.shape[0], 80, 80, 3)
# X_test = X_test.reshape(X_test.shape[0], 80, 80, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# # normalizing the data to help with the training
# X_train /= 255
# X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
'''convert the y train/test to number matrix... idk'''
from sklearn.preprocessing import LabelEncoder
import numpy as np

code = np.array(y_train)
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(y_train)
code2 = np.array(y_test)
label_encoder = LabelEncoder()
vec1 = label_encoder.fit_transform(y_test)
Y_train = np_utils.to_categorical(vec)
Y_test = np_utils.to_categorical(vec1)
# Y_train = y_train
# Y_test = y_test

print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=1, padding='same', activation='relu', input_shape=(80,80, 3)))
# model.add(Conv2D(50, kernel_size=(3,3), activation='relu', input_shape=(80,80, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides= 1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.15))
# output layer sigmoid is for binary classification // softmax for multi classification
model.add(Dense(2, activation='sigmoid')) #int is number of classes (https://stackoverflow.com/questions/48851558/tensorflow-estimator-valueerror-logits-and-labels-must-have-the-same-shape)

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
history = model.fit(X_train, Y_train, batch_size=32, epochs=15, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
joblib.dump(history, "rf_model.pkl")
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()
plt.show()
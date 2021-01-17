from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
from PIL import Image, ImageOps
import numpy as np
import os
from numpy import mean
from scipy.special import softmax
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten ,Dense,Dropout,Softmax

from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import statistics 





# load train and test dataset
X = []
trainY =[]
x = []
testY =[]
i = 0
# ch=['1','2','3','4','5','6','7','8','9','D','G','H','L','M','N','S','T','Y','GH','TA','Y','M',
#                 'N','T','B','SA','J','E','V']
ch=['1','2','3','4','5','6','7','8','9']


test = []

X = []
trainY =[]
x = []
testY =[]
s = []

setnum = 'Original'
epnum = 2
cnntype = 1

num_classes = 9


# scores = open("./Accuracy.txt","a")
# scores.write('\n\n---------------------------------------\n'+setnum+'\nCNN Type '+str(cnntype)+' Epochs: '+str(epnum)+'\n') 

# newpath = './Confusion Matrix/'+setnum+'/CNN/'+str(cnntype)
# if not os.path.exists(newpath):
#     os.makedirs(newpath)

for c in ch:
    path = './Train/'+setnum+'/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 1)
            pix = np.array(im)
            X.append(pix)
            trainY.append(int(c) -1)

for c in ch:
    path = './Test/'+setnum+'/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 1)
            pix = np.array(im)
            x.append(pix)
            testY.append(int(c)-1)

X = np.array(X)
nsamples, nx, ny = X.shape
trainX = X.reshape(nsamples, nx, ny, 1)

x = np.array(x)
nsamples, nx, ny = x.shape
testX = x.reshape(nsamples, nx, ny, 1)


trainY = np.array(trainY)
trainY = tf.keras.utils.to_categorical(trainY , num_classes=9)
# print(trainY.shape)
# exit(1)
testY = np.array(testY)
testY = tf.keras.utils.to_categorical(testY , num_classes=9)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=trainX.shape[1:4]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

if ( cnntype == 0):
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9))

elif( cnntype == 1):
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9))
    model.add(Softmax())



model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

model.fit(trainX, trainY, epochs =epnum ,validation_data=(testX, testY))

score = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))

# acc = [[0 for x in range(9)] for y in range(9)] 
# p = model.predict_classes(testX)
# p_2 = model.predict_proba(testX)
# x_temp = p_2[0 , :]
# print( softmax(x_temp))
model.save('models/')
print(model.evaluate(testX))
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
from PIL import Image, ImageOps
import numpy as np
import os
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pylab as plt
import seaborn as sns




# load train and test dataset
X = []
trainY =[]
x = []
testY =[]
i = 0
# ch=['1','2','3','4','5','6','7','8','9','D','G','H','L','M','N','S','T','Y','GH','TA','Y','M',
#                 'N','T','B','SA','J','E','V']
ch=['1','2','3','4','5','6','7','8','9']

def sw(argument): 
    switcher = { 
        # 'D':'د',
        # 'H': 'ح', 
        # 'L': 'ل',
        # 'GH':'ق',
        # 'S':'س',
        # 'TA':'ط',
        # 'Y':'ی',
        # 'M':'م',
        # 'N':'ن',
        # 'T':'ت',
        # 'B':'ب',
        # 'SA':'ص',
        # 'G':'گ',
        # 'J':'ج',
        # 'E':'ع',
        # 'V':'و',
        '1': "1", 
        '2': "2", 
        '3': "3", 
        '4': "4", 
        '5': "5", 
        '6': "6", 
        '7': "7", 
        '8': "8", 
        '9': "9"
} 
    return switcher.get(argument, "") 

nb_classes = 25
test = []
FontList = [ f for f in os.listdir('./../../Fonts/') ]

for f in FontList:
    X = []
    trainY =[]
    x = []
    testY =[]
    if(f!='.DS_Store'):
        print(f[:-4])
        for c in ch:
            path = './Train/'+f[:-4]+'/'+ c +'/'
            for filename in os.listdir(path):
                if(filename!='.DS_Store'):
                    im = Image.open(path+filename)
                    im = im.convert('L')
                    im = im.point(lambda x: 0 if x<128 else 255)
                    pix = np.array(im)
                    X.append(pix)
                    trainY.append(int(c))

        for c in ch:
            path = './Test/'+f[:-4]+'/'+ c +'/'
            for filename in os.listdir(path):
                if(filename!='.DS_Store'):
                    im = Image.open(path+filename)
                    im = im.convert('L')
                    im = im.point(lambda x: 0 if x<128 else 255)
                    pix = np.array(im)
                    x.append(pix)
                    testY.append(int(c))

        X = np.array(X)
        nsamples, nx, ny = X.shape
        #trainX = X.reshape((nsamples,nx*ny))
        trainX = X.reshape(nsamples, nx, ny, 1)

        x = np.array(x)
        nsamples, nx, ny = x.shape
        #testX = x.reshape((nsamples,nx*ny))
        testX = x.reshape(nsamples, nx, ny, 1)


        trainY = np.array(trainY)
        testY = np.array(testY)


        # label_encoder = LabelEncoder()
        # label_encoder.fit(trainY)
        # trainY = label_encoder.transform(trainY)
        # testY = label_encoder.transform(testY)




        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=trainX.shape[1:4]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(25))


        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])


        model.fit(trainX, trainY, epochs =1 ,validation_data=(testX, testY))

        score = model.evaluate(testX, testY, verbose=0)

        print("Accuracy: %.2f%%" % (score[1]*100))
        
        p = []
        p = model.predict_classes(testX)
        # p = label_encoder.inverse_transform(p)
        # y = label_encoder.inverse_transform(testY)

        # i = 0 
        # for (tl, l) in zip(y, p):
        #     if (l != tl):
        #         print("Label", sw(tl) ,"Pred:", sw(l))
        #         i=i+1
        # print (i)

        print(confusion_matrix(testY,p))

        cmap = sns.diverging_palette(260, 200,l= 90,s=60 ,as_cmap=True,sep=120)
        sns.heatmap(confusion_matrix(testY,p), annot=True, fmt="d",cmap=cmap,linewidths=.5,cbar=1,xticklabels = ch, yticklabels=ch)
        plt.title('Font:'+f[:-4]+' Accuracy: %.2f%%' % (score[1]*100))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('./Confusion Matrix/'+f[:-4]+'.png')
        plt.clf()  # Clear the figure for the next loop

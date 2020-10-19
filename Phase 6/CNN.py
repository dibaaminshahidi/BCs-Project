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


nb_classes = 25
test = []

X = []
trainY =[]
x = []
testY =[]
s = []

setnum = '6'
epnum = 2

scores = open("./Accuracy.txt","a")
scores.write('\n---------------------------------------\nCNN '+str(epnum)+'\n') 

newpath = './Confusion Matrix/'+setnum+'/CNN'
if not os.path.exists(newpath):
    os.makedirs(newpath)

for c in ch:
    path = './Train/'+setnum+'/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 255)
            pix = np.array(im)
            X.append(pix)
            trainY.append(int(c))

for c in ch:
    path = './Test/'+setnum+'/'+ c +'/'
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
trainX = X.reshape(nsamples, nx, ny, 1)

x = np.array(x)
nsamples, nx, ny = x.shape
testX = x.reshape(nsamples, nx, ny, 1)


trainY = np.array(trainY)
testY = np.array(testY)




for i in range(6):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=trainX.shape[1:4]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(25))


    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(trainX, trainY, epochs =epnum ,validation_data=(testX, testY))

    score = model.evaluate(testX, testY, verbose=0)
    print("Accuracy: %.2f%%" % (score[1]*100))

    acc = [[0 for x in range(9)] for y in range(9)] 
    p = model.predict_classes(testX)

    cm = confusion_matrix(testY,p)
    print(cm)

    cmap = sns.diverging_palette(260, 200,l= 90,s=60 ,as_cmap=True,sep=120)
    sns.heatmap(cm, annot=True, fmt="d",cmap=cmap,linewidths=.5,cbar=0,xticklabels = ch, yticklabels=ch)
    plt.title(' Accuracy: %.2f%%' % (score[1]*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+setnum+'/CNN/'+str(i+1)+' CM.png')
    plt.clf()  # Clear the figure for the next loop
    acc = cm/cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10,10))   
    sns.heatmap(acc,square=True ,annot=True,cmap=cmap,linewidths=.5,cbar=0,xticklabels = ch, yticklabels=ch)
    plt.title('Accuracy: %.2f%%' % (score[1]*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+setnum+'/CNN/'+str(i+1)+' Ratio.png')
    plt.clf()  # Clear the figure for the next loop
    scores.write('Accuracy: %.2f%% ' % (score[1]*100))
    s.append(score[1])
avrg = statistics.mean(s)*100
scores.write('\n'+ 'CNN '+ str(epnum)+'Average '+setnum+': %.2f%% \n' %(avrg)) 
scores.close()


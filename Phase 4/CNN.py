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

X = []
trainY =[]
x = []
testY =[]
s = []


scores = open("./Accuracy.txt","a")#append mode 


with open("Names.txt", "r") as file:
    first_line = file.readline()
first_line
scores.write('\n'+first_line+'\n') 

newpath = './Confusion Matrix/'+ first_line
if not os.path.exists(newpath):
    os.makedirs(newpath)
#     print('hey')
# print('hi')

for c in ch:
    path = './Train/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 255)
            pix = np.array(im)
            X.append(pix)
            trainY.append(int(c))

for c in ch:
    path = './Test/'+ c +'/'
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



for i in range(6):
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

    model.fit(trainX, trainY, epochs =2 ,validation_data=(testX, testY))

    score = model.evaluate(testX, testY, verbose=0)
    print("Accuracy: %.2f%%" % (score[1]*100))

    acc = [[0 for x in range(9)] for y in range(9)] 
    p = model.predict_classes(testX)
    # p = label_encoder.inverse_transform(p)
    # y = label_encoder.inverse_transform(testY)

    # i = 0 
    # for (tl, l) in zip(y, p):
    #     if (l != tl):
    #         print("Label", sw(tl) ,"Pred:", sw(l))
    #         i=i+1
    # print (i)
    cm = confusion_matrix(testY,p)
    #print(cm)

    cmap = sns.diverging_palette(260, 200,l= 90,s=60 ,as_cmap=True,sep=120)
    sns.heatmap(cm, annot=True, fmt="d",cmap=cmap,linewidths=.5,cbar=0,xticklabels = ch, yticklabels=ch)
    plt.title(str(i)+' Accuracy: %.2f%%' % (score[1]*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+first_line+'/'+first_line+' '+str(i)+' '+str(score[1]*100)+' CM.png')
    plt.clf()  # Clear the figure for the next loop
    acc = cm/cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10,10))   
    sns.heatmap(acc,square=True ,annot=True,cmap=cmap,linewidths=.5,cbar=0,xticklabels = ch, yticklabels=ch)
    plt.title('Accuracy: %.2f%%' % (score[1]*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+first_line+'/'+first_line+' '+str(i)+' '+str(score[1]*100)+' Ratio.png')
    plt.clf()  # Clear the figure for the next loop
    scores.write('Accuracy: %.2f%% ' % (score[1]*100))
    s.append(score[1])
    print(s)
avrg = statistics.mean(s)*100
scores.write('\n'+first_line + ' Average: %.2f%% \n' %(avrg)) 
scores.close()


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
from tensorflow import keras

from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import statistics 


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def fitness_for_a_pic(pic , target_number):
    model = keras.models.load_model('models')
    y = [target_number for _ in range(pic.shape[0])]
    y = np.array(y)
    y = tf.keras.utils.to_categorical(y , num_classes=9)
    per = model.predict_proba(pic)
    print(per.shape)
    print(y.shape)
    cer = []
    for i in range(y.shape[0]):
        qone = per[i,:]
        qone = np.expand_dims(qone , 0)
        qtwo = y[i,:]
        qtwo = np.expand_dims(qtwo , 0)
        # print(cross_entropy(qone , qtwo))
        cer.append(-1 * cross_entropy(qone , qtwo))
        # print(qone.shape)
        # print(qtwo.shape)
    # print(pic.shape)
    # pic = np.expand_dims(pic,0)
    # input shape for pic = (50 , 50 ,1)
    # vals = model.evaluate(pic , y)
    # print(vals)
    # exit(1)
    return cer



if __name__ == '__main__':

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



    x = np.array(x)
    nsamples, nx, ny = x.shape
    testX = x.reshape(nsamples, nx, ny, 1)
    testY = np.array(testY)



    pic = testX[0]
    print(fitness_for_a_pic(pic , testY[0]))



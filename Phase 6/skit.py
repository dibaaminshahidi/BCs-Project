from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
from PIL import Image, ImageOps
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
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


nb_classes = 25
test = []

X = []
trainY =[]
x = []
testY =[]
s = []

setnum = 'Original'


scores = open("./Accuracy.txt","a")
newpath = './Confusion Matrix/'+setnum
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
d2_train_dataset = X.reshape((nsamples,nx*ny))

x = np.array(x)
nsamples, nx, ny = x.shape
d2_test_dataset = x.reshape((nsamples,nx*ny))

classifiers = [
RandomForestClassifier(random_state=0,max_depth = 5,n_estimators = 200),
svm.SVC()
        ]

for clf in classifiers:
    scores.write('\n---------------------------------------\n6\n') 
    scores.write(clf.__class__.__name__+'\n') 
    new = './Confusion Matrix/'+setnum+'/'+clf.__class__.__name__
    if not os.path.exists(new):
        os.makedirs(new)

    clf.fit(d2_train_dataset,trainY)
    p = clf.predict(d2_test_dataset)
    score = accuracy_score(testY, p)
    print("Accuracy: %.2f%%" % (score*100))
    #print("Overfit: %.2f%%" % (accuracy_score(Y, clf.predict(d2_train_dataset))*100))
    # for (tl, l) in zip(y, p):
    #     if (l != tl):
    #         print("Label", sw(tl) ,"Pred:", sw(l))
    cm =confusion_matrix(testY,p)
    cmap = sns.diverging_palette(260, 200,l= 90,s=60 ,as_cmap=True,sep=120)
    sns.heatmap(cm, annot=True, fmt="d",cmap=cmap,linewidths=.5,cbar=1,xticklabels = ch, yticklabels=ch)
    plt.title('Accuracy: %.2f%%' % (score*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+setnum+'/'+clf.__class__.__name__+'/'+str(i+1)+' CM.png')
    plt.clf()  # Clear the figure for the next loop
    acc = cm/cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10,10))   
    sns.heatmap(acc,square=True ,annot=True,cmap=cmap,linewidths=.5,cbar=0,xticklabels = ch, yticklabels=ch)
    plt.title('Accuracy: %.2f%%' % (score*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./Confusion Matrix/'+setnum+'/'+clf.__class__.__name__+'/'+str(i+1)+' Ratio.png')
    plt.clf()  # Clear the figure for the next loop
    scores.write('Accuracy: '+setnum+': %.2f%% ' % (score*100))
    scores.write('\nAverage '+setnum+': %.2f%% \n' %(avrg)) 

scores.close()


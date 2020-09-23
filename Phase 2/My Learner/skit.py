
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
from PIL import Image, ImageOps
import numpy as np
from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
import os
from sklearn.metrics import confusion_matrix


X = []
Y =[]
x = []
y =[]
i = 0
ch=['1','2','3','4','5','6','7','8','9','D','G','H','L','M','N','S','T','Y','GH','TA','Y','M',
                'N','T','B','SA','J','E','V']

for c in ch:
    path = './Train/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 255)
            pix = np.array(im)
            X.append(pix)
            Y.append(c)

for c in ch:
    path = './Test/'+ c +'/'
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 255)
            pix = np.array(im)
            x.append(pix)
            y.append(c)
print(len(Y), len(X),len(y), len(x))


# clf = MLPClassifier(solver='lbfgs',activation  ='tanh', 
#                       alpha=1e-5, hidden_layer_sizes=(10, 12), random_state=30,learning_rate='adaptive')

#clf = RandomForestClassifier()

clf = svm.SVC()

X = np.array(X)
nsamples, nx, ny = X.shape
d2_train_dataset = X.reshape((nsamples,nx*ny))
clf.fit(d2_train_dataset,Y)

x = np.array(x)
nsamples, nx, ny = x.shape
d2_test_dataset = x.reshape((nsamples,nx*ny))
p = clf.predict(d2_test_dataset)
print(accuracy_score(y, p))
for (tl, l) in zip(y, p):
    if (l != tl):
        print("Label", tl ,"Pred:", l)

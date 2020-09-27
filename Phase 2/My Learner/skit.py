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






X = []
Y =[]
x = []
y =[]
i = 0
ch=['1','2','3','4','5','6','7','8','9','D','G','H','L','M','N','S','T','Y','GH','TA','Y','M',
                'N','T','B','SA','J','E','V']
def sw(argument): 
    switcher = { 
        'D':'د',
        'H': 'ح', 
        'L': 'ل',
        'GH':'ق',
        'S':'س',
        'TA':'ط',
        'Y':'ی',
        'M':'م',
        'N':'ن',
        'T':'ت',
        'B':'ب',
        'SA':'ص',
        'G':'گ',
        'J':'ج',
        'E':'ع',
        'V':'و',
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
test = []
for c in ch:
    path = './Test/'+ c +'/'
    n = 0
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 255)
            pix = np.array(im)
            x.append(pix)
            y.append(c)
            n=n+1
    test.append(c)
    test.append(n)
print(test)
#clf = MLPClassifier(solver='lbfgs',activation  ='tanh', 
#                alpha=1e-5, hidden_layer_sizes=(10, 12), random_state=30,learning_rate='adaptive')
#clf = RandomForestClassifier()
#clf = svm.SVC()

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
    print('\n',clf.__class__.__name__)
    clf.fit(d2_train_dataset,Y)
    p = clf.predict(d2_test_dataset)
    print("Accuracy: %.2f%%" % (accuracy_score(y, p)*100))
    #print("Overfit: %.2f%%" % (accuracy_score(Y, clf.predict(d2_train_dataset))*100))
    for (tl, l) in zip(y, p):
        if (l != tl):
            print("Label", sw(tl) ,"Pred:", sw(l))


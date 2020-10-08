from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
from PIL import Image, ImageOps
from justifytext import justify
import persian
import random
import cv2
import numpy as np
import os

import scipy.sparse as sparse
import scipy.stats as stats

 
#Generate 

def sw(argument): 
    switcher = { 
        # 'د': "D", 
        # 'ح': "H", 
        # 'ل': "L", 
        # 'ق': "GH", 
        # 'س': "S", 
        # 'ط': "TA", 
        # 'ی': "Y", 
        # 'م': "M", 
        # 'ن': "N", 
        # 'ت': "T", 
        # 'ب': "B", 
        # 'ص': "SA", 
        # 'گ': "G", 
        # 'ج': "J", 
        # 'ع': "E", 
        # 'و': "V", 
        1: "1", 
        2: "2", 
        3: "3", 
        4: "4", 
        5: "5", 
        6: "6", 
        7: "7", 
        8: "8", 
        9: "9", 

} 

    return switcher.get(argument, "") 

b = 1


n = 900

names = ["  " for x in range(10)]
names =np.array(names)

for x in range(n):

    r = random.randint(1,9)
    TrainPath = './Train/'+ sw(r) + '/'
    TestPath = './Test/'+sw(r) + '/'


    if(b==1):
        img = Image.open("./../Assets/Blank.jpg")
        new_size = (75,75)
        img = img.resize(new_size)

        number = persian.enToPersianNumb(r)
        draw = ImageDraw.Draw(img)
        W, H = img.size
        fontsize = random.randint(45,70)  


        if(r==3): 
            f = 'Traffic.ttf'
            names[3]=f[:-4]


        elif(r==2): 
            f = 'Farnaz.ttf'
            names[2]= f[:-4]

        elif(r==6): 
            f = 'Morvarid.ttf'
            names[6]= f[:-4]
        elif(r==9): 
            f = 'Homa.ttf'
            names[9]= f[:-4]
        else:
            f = 'STITRBD.ttf'
        font = ImageFont.truetype('./../Fonts/'+ f ,fontsize)
        #font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
        w, h = font.getsize(number)
        draw.text(((W-w)/2,((H-h)/2)),number,(10,10,10),font=font,align="center")

        #BLUR
        #img = img.filter(ImageFilter.BLUR)
    
        #SALT AND PEPPER 
        # np.random.seed(42)
        # snp = sparse.random(H, W, density=0.3)*255
        # snp = snp.toarray()
        # snp = Image.fromarray(snp.astype('uint8')).convert('L')
        # mask = Image.new("L", img.size,150)
        # img = Image.composite(img, snp , mask)

        #Gaussian
        img = img.filter(ImageFilter.GaussianBlur(radius = 2)) 

        
        #Perspective
        p = 4
        for z in range(p):
            new_img = img
            width, height = img.size
            a = random.uniform(-0.2,0.2)
            xshift = abs(a) * width
            new_width = width + int(round(xshift))
            new_img = new_img.transform((new_width, height), Image.AFFINE,
                    (1, a, -xshift if a > 0 else 0, 0, 1, 0), Image.BICUBIC)
            A = str(a)

            #Rotate
            rand = random.randint(-5,5)
            R = str(rand)
            new_img = new_img.rotate(rand)

            new_img = new_img.crop((15,8,67,60))
            new_size = (80,80)
            new_img = new_img.resize(new_size)
            new_size = (50,50)
            new_img = new_img.resize(new_size)

            #Save
            c = random.randint(1,4)
            if(c!=1):
                new_img.save(TrainPath + A + R + '.jpg')
            else:
                new_img.save(TestPath + A + R + '.jpg')

    if(b==0):
        filelist = [ d for d in os.listdir(TrainPath) ]
        for d in filelist:
            os.remove(os.path.join(TrainPath, d))  
        filelist = [ d for d in os.listdir(TestPath) ]
        for d in filelist:
            os.remove(os.path.join(TestPath, d)) 

s = ''           
for x in range (len(names)):
    if names[x]!="  ":
        s = s + str(x) + names[x]

# s = 'Orginal'
fl = open("./Names.txt", "w")
fl.write(s)
fl.close()
    

    





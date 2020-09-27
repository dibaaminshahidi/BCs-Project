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


 

n = 1000


for x in range(n):

    img = Image.open("./../Assets/Blank.jpg")
    new_size = (75,75)
    img = img.resize(new_size)
    # r = random.randint(1,3)
    # if (r==1):
    r = random.randint(1,9)
    TrainPath = './../Phase 2/My Learner/NumTrain/' + sw(r) + '/'
    TestPath = './../Phase 2/My Learner/NumTest/' + sw(r) + '/' 
    number = persian.enToPersianNumb(r)
    # else :
    #     number = random.choice(['ع', 'و', 'ج', 'ت', 'ص', 'ب','ن', 'م', 'ی', 'د', 'ح', 'ل', 'ق', 'س', 'ط','گ'])
    #     #number='ق'
    #     TrainPath = './../Phase 2/My Learner/Train/' + sw(number) + '/' 
    #     TestPath = './../Phase 2/My Learner/Test/' + sw(number) + '/' 

        


    draw = ImageDraw.Draw(img)
    W, H = img.size
    fontsize = random.randint(30,60)   
    font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
    w, h = font.getsize(number)
    draw.text(((W-w)/2,((H-h)/2)),number,(10,10,10),font=font,align="center")

    #BLUR
    img = img.filter(ImageFilter.BLUR)
 
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


            
        
        #b=0
        b=1
        rand = random.randint(-5,5)
        R = str(rand)
        new_img = new_img.rotate(rand)
        # new_size = (30,30)
        # new_img = new_img.resize(new_size)
        new_img = new_img.crop((15,15,60,60))
        new_size = (50,50)
        new_img = new_img.resize(new_size)

        if(b==1):
            c = random.randint(1,4)
            if(c != 1):
                new_img.save(TrainPath + A + R + '.jpg')
            else:               
                new_img.save(TestPath + A + R + '.jpg')

    if(b==0):
        filelist = [ f for f in os.listdir(TrainPath) ]
        for f in filelist:
            os.remove(os.path.join(TrainPath, f))  
        filelist = [ f for f in os.listdir(TestPath) ]
        for f in filelist:
            os.remove(os.path.join(TestPath, f))  


            





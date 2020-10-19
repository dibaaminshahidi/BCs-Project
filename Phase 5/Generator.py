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


b = 1
setnum = '6'

for x in range(9):


    TrainPath = './Train/'+ str(x+1) + '/'
    TestPath = './Test/'+ str(x+1) + '/'
    r = random.randint(100,131)
    for y in range(r):
        if(b==1):
            #img = Image.open("./../Assets/Blank.jpg")
            inpath = './NUMBERS/'+setnum+'/Font_'+str(x+1)+'.jpg'
            img = Image.open(inpath)        
            width, height = img.size 
            rndsize = random.randint(100,181)
            left = (width - rndsize)/2
            right = (width + rndsize)/2
            bottom = ((height + rndsize)/2)
            top = ((height - rndsize)/2)

            img = img.crop((left, top, right, bottom))
            new_size = (300,300)
            img = img.resize(new_size)
    
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
            img = img.filter(ImageFilter.GaussianBlur(radius = 4)) 

            #Perspective
            p = 4
            new_size = (75,75)
            img = img.resize(new_size)
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
                left = (width - 52)/2
                top = (height - 52)/2
                right = (width + 52)/2
                bottom = (height + 52)/2
                new_img = new_img.crop((left, top, right, bottom))
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

    

    





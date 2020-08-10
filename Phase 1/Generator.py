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
 
#Generate 

n = 10

for x in range(n):
    img = Image.open("./../Assets/Raw.jpg")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    r = 0
    number = ""
    left = W
    top = H
    right = W
    bottom = H

    for y in range(0,6):
        if (y==2):
            # r = random.choice(['ط', 'د','س', 'ش', 'ک', 'ن', 'ل', 'ق', 'ص', 'ت', 'ب'])
            r = random.choice([ 'ن', 'ل', 'ق', 'ص', 'ت'])
            l = r
            number = number + r
        else:
            r = random.randint(1,9)
            number = number + persian.enToPersianNumb(r)
     
    iran = "۱ ۱"
    fontsize = 85
    font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
    draw.text((W-110, (H/2)-20),iran,(10,10,10),font=font,align="center")
    
    fontsize = 95
    font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
    w, h = font.getsize(number)
    d = 33
    #print(type(l))
    #print(l)
    t , h = font.getsize(l)
    if(h > 80):
        fontsize = fontsize-1
        d = 40
    while(w>330):
        fontsize = fontsize-1
        font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
        w, h = font.getsize(number)
        t, h = font.getsize(l)
        if(h > 80):
            d = 40
        # print(h)


    draw.text((60, (H/2)-d),number,(10,10,10),font=font,align='center')
    
    num = str(x)
    img.save('./Output/Simple/Resault '+ num +'.jpg')


    #BLUR
    img1 = img.filter(ImageFilter.BLUR)
    img1.save('./Output/Noisy/Blur/Blur '+ num +'.jpg')
 
    #SALT AND PEPPER 
    m = 4
    for y in range(m):
        img2 = img1
        snp = np.random.rand(H,W,3) * 255
        snp = Image.fromarray(snp.astype('uint8')).convert('L')
        mask = Image.new("L", img1.size,128)
        img2 = Image.composite(img2, snp , mask)
        Y = str(y)
        img2.save('./Output/Noisy/Salt & Pepper/SnP '+ Y + num +'.jpg')

        #Perspective
        p = 10
        for z in range(p):
            width, height = img2.size
            a = random.uniform(-1,1)
            xshift = abs(a) * width
            new_width = width + int(round(xshift))
            img4 = img2.transform((new_width, height), Image.AFFINE,
                    (1, a, -xshift if a > 0 else 0, 0, 1, 0), Image.BICUBIC)
            Z = str(z)
            img4.save('./Output/Perspective/Perspective '+ Z + Y + num +'.jpg')

            #Rotate
            n = random.randint(-45,45)
            delta_w = W
            delta_h = 3*H
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            # print(padding)
            new_im = ImageOps.expand(img4, padding, fill="black")
            rotated = new_im.rotate(n)

            rotated.save('./Output/Rotated/Rotated '+ Z + Y + num +'.jpg')


            





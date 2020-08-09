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

n = 20

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
            r = random.choice(['ط', 'د','س', 'ش', 'ک', 'ن', 'ل', 'ق', 'ص', 'ت', 'ب'])
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

    # print(w)

    while(w>330):
        fontsize = fontsize-1
        font = ImageFont.truetype("./../Fonts/STITRBD.TTF",fontsize)
        w, h = font.getsize(number)

    draw.text((60, (H/2)-33),number,(10,10,10),font=font,align='center')
    
    num = str(x)
    img.save('./Output/Simple/Resault '+ num +'.jpg')


    #BLUR
    im1 = img.filter(ImageFilter.BLUR)
    im1.save('./Output/Noisy/Blur/Blur '+ num +'.jpg')
 
    #SALT AND PEPPER

    m = 4

    for y in range(m):
        im2 = img
        snp = np.random.rand(H,W,3) * 255
        snp = Image.fromarray(snp.astype('uint8')).convert('L')
        mask = Image.new("L", im1.size,128)
        im2 = Image.composite(im2, snp , mask)
        Y = str(y)
        im2.save('./Output/Noisy/Salt & Pepper/SnP '+ Y + num +'.jpg')

        # BLUR AND SALT AND PEPPER
        im3 = im2.filter(ImageFilter.BLUR)
        im3.save('./Output/Noisy/Blur with Salt & Pepper/Mixed '+ Y + num +'.jpg')
        
        p = 10
        
        for z in range(p):
            #Perspective
            width, height = im3.size
            a = random.uniform(-0.5,0.5)
            xshift = abs(a) * width
            new_width = width + int(round(xshift))
            im4 = im3.transform((new_width, height), Image.AFFINE,
                    (1, a, -xshift if a > 0 else 0, 0, 1, 0), Image.BICUBIC)
            Z = str(z)
            im4.save('./Output/Perspective/Perspective '+ Z + Y + num +'.jpg')

            #Rotate
            n = random.randint(-45,45)
            delta_w = W
            delta_h = 3*H
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            # print(padding)
            new_im = ImageOps.expand(im4, padding, fill="black")
            rotated = new_im.rotate(n)

            rotated.save('./Output/Rotated/Rotated '+ Z + Y + num +'.jpg')


            





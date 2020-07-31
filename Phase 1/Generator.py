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
for x in range(10):
    img = Image.open("./../Assets/Raw.jpg")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    n = 0
    number = ""
    left = W
    top = H
    right = W
    bottom = H

    for y in range(0,6):
        if (y==2):
            n = "ب"
            number = number + n
        else:
            n = random.randint(1,9)
            number = number + persian.enToPersianNumb(n)
     
    iran = "۱ ۱"
    fontsize = 95
    font = ImageFont.truetype("./../Fonts/B Traffic Bold.ttf",fontsize)
    w, h = font.getsize(number)
    while(w>325):
        fontsize = fontsize-1
        font = ImageFont.truetype("./../Fonts/B Traffic Bold.ttf",fontsize)
        w, h = font.getsize(number)
    draw.text((60, (H/2)-33),number,(10,10,10),font=font,align='center')
    draw.text((W-110, (H/2)-23),iran,(10,10,10),font=font,align="center")
    num = str(x)
    img.save('./Output/Simple/Resault'+ num +'.jpg')

    #BLUR
    im1 = img.filter(ImageFilter.BLUR)
    im1.save('./Output/Noisy/Blur'+ num +'.jpg')
 
    #SALT AND PEPPER
    for y in range(4):
        im2 = img
        snp = np.random.rand(H,W,3) * 255
        snp = Image.fromarray(snp.astype('uint8')).convert('L')
        mask = Image.new("L", im1.size,128)
        im2 = Image.composite(im2, snp , mask)
        Y = str(y)
        im2.save('./Output/Noisy/Salt & Pepper'+ Y + num +'.jpg')

        # BLUR AND SALT AND PEPPER
        im3 = im2.filter(ImageFilter.BLUR)
        im3.save('./Output/Noisy/Blur with Salt & Pepper'+ Y + num +'.jpg')
        
        for z in range(10):
            n = random.randint(-45,45)

            delta_w = 3*W 
            delta_h = 3*H
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            print(padding)
            new_im = ImageOps.expand(im3, padding, fill="black")
            rotated = new_im.rotate(n)
            Z = str(z)
            rotated.save('./Output/Rotated/Rotated'+ Z + Y + num +'.jpg')






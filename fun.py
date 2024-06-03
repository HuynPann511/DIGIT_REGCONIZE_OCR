import easyocr 
import cv2
import numpy as np

import matplotlib.pyplot as plt
import time
import urllib.request
url = 'http://192.168.137.108/cam-hi.jpg'
while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    image = cv2.imdecode(imgnp,-1)
    image = cv2.medianBlur(image,7)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #to gray convert
    th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2) #adaptive threshold gaussian filter used
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
    

    x = 0   #to save the position, width and height for contours(later used)
    y = 0
    w = 0
    h = 0

    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    threshold =  10
    font = cv2.FONT_HERSHEY_SIMPLEX  
    org = (50, 50) 
    fontScale = 1 
    color = (0, 0, 0)
    thickness = 2
        
    for c in cnts:
        
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c)   
        if  len(approx) == 4 and area > 100000:   #manual area value used to find ROI for rectangular contours
        
            cv2.drawContours(image,[c], 0, (0,255,0), 3)
            n = approx.ravel()
            font = cv2.FONT_HERSHEY_SIMPLEX
            (x, y, w, h) = cv2.boundingRect(c)
            old_img = opening[y:y+h, x:x+w]  #selecting the ROI
            width, height = old_img.shape
            cropped_img = old_img[50:int(width/2), 0:height] #cropping half of the frame of ROI to just focus on the number
            
            new = reader.readtext(cropped_img)   #reading text using easyocr
            if(new == []): 
                text = 'none'
            else:
                text = new
                print(text)
#                 cv2.rectangle(cropped_img, tuple(text[0][0][0]), tuple(text[0][0][2]), (0, 0, 0), 2)
                if(text[0][2] > 0.5): #checking the confidence level
                    
                    cv2.putText(cropped_img, text[0][1], org, font, fontScale, color, thickness, cv2.LINE_AA)        
            cv2.imshow('frame1',cropped_img)
    key = cv2.waitKey(5) 

    if key == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
    
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
count=0;
#img = cv2.imread('chess.jpg')
cap=cv2.VideoCapture('prueba1_out.mp4')
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
back=cv2.bgsegm.createBackgroundSubtractorMOG()
cv2.ocl.setUseOpenCL(False)
while (1):
    ret, img=cap.read()
    
    #img=img[100:320,100:240]
    fram=cv2.medianBlur(img,7)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize = 3)
    fgmask=back.apply(fram)
    blur=cv2.GaussianBlur(fgmask,(7,7),1)
    fgmask=cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    
    contor=fgmask.copy()
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(0,105),(320,105),(0,0,255),2)
        cv2.line(img,(0,205),(320,205),(0,0,255),2)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.line(img,(53,y1),(x2-x1-53,y2),(0,0,255),2)
    im, contours,hierarchy =cv2.findContours(contor,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
           if cv2.contourArea(c) >=400:
             #cv2.drawContours(frame,contours,-1,(0,255,0),2)
             x,y,ancho,alto=cv2.boundingRect(c)
             a=int(x+ancho/2)
             b=int(y+alto/2)
                        
             if 105<b<205 :
                cv2.rectangle(img,(x,y),(x+ancho,y+alto),(0,0,255),2)
                print('cruzo en Rojo!!!')
                cv2.imwrite("./imagenes/fram%d.jpg" %count,img)
                count+=1
             else:
                cv2.rectangle(img,(x,y),(x+ancho,y+alto),(0,255,0),2)

            
             
    
   

   # time.sleep(.1)
    cv2.imshow('houghlines3.jpg',img)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()

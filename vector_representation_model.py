import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

class Tag():
    def __init__(self,a,b,index):
        self.a=a
        self.b=b
        self.index=index
        self.vector=[a,b]
    def _print(self):
        print(self.a, self.b,self.index)
    def _draw(self,image):
        self.image=image
        self.vector=cv2.line(self.image,(0,0),(self.a,self.b),(0,255,0),2)
        return self.vector
def comparator(lista):
      for i in range(len(lista)):
          try:
              prevX=lista[i]
              postX=lista[i+1]
              restax=prevX.vector[0]-postX.vector[0]
              restay=prevX.vector[1]-postX.vector[1]
              module=np.sqrt(restax**2+restay**2)
              print(module)
          except Exception as e:
              print('not ready',e)



list_tags=[]
count=0;
num=0;
cap=cv2.VideoCapture('prueba1_out.mp4')
   ##Mascaras e inicio de backgroundSubtractor
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
back=cv2.bgsegm.createBackgroundSubtractorMOG()
cv2.ocl.setUseOpenCL(False)
while (1):
    ret, img=cap.read()
    ##cambio  de tamaño
    img=img[0:320,10:210]
   
    fram=cv2.medianBlur(img,7)
    fgmask=back.apply(fram)
    blur=cv2.GaussianBlur(fgmask,(7,7),1)
    fgmask=cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    contor=fgmask.copy()
    im, contours,hierarchy =cv2.findContours(contor,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
           if cv2.contourArea(c) >=460:
             x,y,ancho,alto=cv2.boundingRect(c)
             a=int(x+ancho/2)
             b=int(y+alto/2)
                        
             if 105<b<205 :
                
                cv2.rectangle(img,(x,y),(x+ancho,y+alto),(0,0,255),2)
                #print('cruzo en Rojo!!!')
                cv2.putText(img, "#={}".format(num),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)
                #cv2.imwrite("./imagenes/fram%d.jpg" %count,img)
                num=1+num
                tag=Tag(a,b,num)
                #tag._print()
                tag._draw(img)
                list_tags.append(tag)
                
                comparator(list_tags)
                count+=1

    time.sleep(.1)
    cv2.imshow('houghlines3.jpg',img)
    #plt.imshow(img, cmap='gray', interpolation='bicubic')
    #plt.xticks([]), plt.yticks([])
    #plt.show()
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()

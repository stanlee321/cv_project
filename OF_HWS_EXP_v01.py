#!/usr/bin/env python

import numpy as np
import cv2
#import cv2.cv as cv

from sklearn.cluster import KMeans
from scipy.ndimage.measurements import label

help_message = '''
USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

'''

cascade_src = 'cars.xml' # LOADING CASCADE XML
car_cascade = cv2.CascadeClassifier(cascade_src)
font = cv2.FONT_HERSHEY_SIMPLEX
        
# Params for shitomasi corner dectection
#feature_params = dict( maxCorners = 10, qualityLevel = 0.3, 
#                      minDistance = 7, blockSize = 7)


# flow sume fx,fy va
total_flow_frame = np.array((0,0))
def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):

    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    #vertices = np.array([[650,0],[750,0],[500,600],[0,600],[0,150],[650,0]], np.int32)
    #vertices = np.array([[140,0],[300,0],[200,231],[0,231],[0,58],[260,0]], np.int32) # for 320*240
    
    #Fur 240x360
    #vertices = np.array([[100,0],[240,0],[150,231],[0,231],[0,150],[230,0]], np.int32)
    
    #Fur 640x480

    vertices = np.array([[450,100],[430,100],[440,460],[0,460],[0,300],[450,100]], np.int32)
    
    processed_img = roi(processed_img, [vertices])
    return processed_img

def draw_flow(img, flow, theta, step=10):
    total_flow_framex = 0
    total_flow_framey = 0
    h, w = img.shape[:2]
    #print(h,w)
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    y = np.int32(y)
    x = np.int32(x)
 
    fx, fy = flow[y,x].T
  

    total_flow_framex += sum(fx)
    total_flow_framey += sum(fy)

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    #print(sum(fx))
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    

    """CASCADE PART """
    #cars = car_cascade.detectMultiScale(vis, 1.1, 1)
    #for (x1,y1,w1,h1) in cars:
    #    cv2.rectangle(vis,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
    """ END CASCADE PART """



    #print(total_flow_framex, total_flow_framey)
    total_flow = np.array([total_flow_framex, total_flow_framey])
    module = np.sqrt(total_flow[0]**2  + total_flow[1]**2)


    unitary_vector = np.array([np.cos(np.pi*theta/180),-np.sin(np.pi*theta/180)])
    scalar_vel = -total_flow[0]*unitary_vector[0] + total_flow[1]*unitary_vector[1]
    vector_vel = scalar_vel * unitary_vector
    module_vector_vel = np.sqrt(vector_vel[0]**2 + vector_vel[1]**2)

    # total_flow_vector draw
    #cv2.line(vis,(120,160),(120+int(sum(fx)),160+int(sum(fy))),(255,0,0),2)
    
    origen = np.array([w//2, h//2])


    #name = str(module_vector_vel)
    name = str(scalar_vel)
    cv2.line(vis,(origen[0],origen[1]),(origen[0]+int(vector_vel[0]),origen[1]+int(vector_vel[1])),(255,0,0),2)
    cv2.putText(vis,name,(origen[0],origen[1]), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
    cv2.polylines(vis, lines, 0, (255, 55, 0))


    return vis, total_flow

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res





source = []

def __main_function__(fn, again):

    cam = cv2.VideoCapture(fn)
    ret, prev = cam.read()

    #is_red = _traffic_light_is_red()
    is_red = True

    
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    #p0 = cv2.goodFeaturesToTrack(prevgray, mask = None, **feature_params)

    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    
    while is_red:
        ret, img = cam.read()
        img = np.array(process_img(img))
        #original_ = img.copy()
        #img= img[0:320,10:240]

        

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.1, 3, 15, 3, 5, 1.2, 0)

        #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None,0.4, 1, 12, 2, 8, 1.2, 0)
        prevgray = gray
        #X = draw_flow(gray, flow)[1]
        #print(X)
        theta = 55
        vis, total_flow = draw_flow(gray, flow, theta)

        #print(total_flow)
        cv2.imshow('flow', vis )
        #p0 = flow.reshape(-1,1,2)
        #p0 = (cv2.goodFeaturesToTrack(prevgray, mask = None, **feature_params)).reshape(-1,1,2)

        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print ('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print( 'glitch is', ['off', 'on'][show_glitch])
    #cv2.destroyAllWindows()

    else:
        print(source[0])
        return __main_function__(source[0], True)
 
if __name__ == '__main__':
    
    import sys
    print (help_message)
    try: fn = sys.argv[1]

    except: fn = 0
    __main_function__(fn, False)

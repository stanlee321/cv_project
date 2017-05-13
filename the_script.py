import YOLO_tiny_tf
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import os

yolo = YOLO_tiny_tf.YOLO_TF()



dirname = '/home/stanlee321/Desktop/proyect/YOLO_tensorflow/cars_brad'
output_img =  '/home/stanlee321/Desktop/proyect/YOLO_tensorflow/outputs_img/'
outputs_logs =  '/home/stanlee321/Desktop/proyect/YOLO_tensorflow/outputs_logs/'

																																																																																																																																																																																										
files = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]

i=0
for image in files:

	yolo.disp_console = True #(True or False, default = True)
	yolo.imshow = True #(True or False, default = True)
	yolo.tofile_img = output_img+'{}.jpg'.format(i) #(output image filename)
	yolo.tofile_txt = outputs_logs+'logs_{}.txt'.format(i) #(output txt filename)
	yolo.filewrite_img = True #(True or False, default = False)
	yolo.filewrite_txt =  True #(True of False, default = False)


	filename = image
	yolo.detect_from_file(filename)
	#yolo.detect_from_cvmat(cvmat)
	try:
																																																																																																																																																																																																																																																																																																															
		yolo.detect_from_file(filename)
	except Expection as e:
		print('I cant this image : ', image)
	#yolo.detect_from_cvmat(cvmat)
	i+=1



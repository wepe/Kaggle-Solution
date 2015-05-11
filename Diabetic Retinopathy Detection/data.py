#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe

"""


import os
from PIL import Image
import numpy as np
import csv

def get_trainlabel():
	table = {}
	rows = csv.reader(open( '../trainLabels.csv','rb'))
	rows.next()
	for row in rows:
		table[row[0]] = int(row[1])
	return table

def load_color_img(direction="./trian_RGB64_plus/"):
	imgs = os.listdir(direction)
	num = len(imgs)
	table = get_trainlabel()
	traindata = np.empty((num,3,64,64),dtype="float32")
	trainlabel = np.empty((num,),dtype="uint8")
	for i in range(num):
		imgname = imgs[i]
		img = Image.open(direction+imgname)
		arr = np.asarray(img,dtype="float32")
		traindata[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		trainlabel[i] = table[imgname.split('.')[0]]
	return traindata,trainlabel









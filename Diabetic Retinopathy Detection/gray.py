#*-*coding:utf8*-*#

#转化为灰度图128*128

import os
from PIL import Image

direction = "/home/wepon/DR/train_smallsize"
save_path = "/home/wepon/DR/gray128"


if not os.path.exists(save_path):
	os.mkdir(save_path)

os.chdir(save_path)
imglist = os.listdir(direction)
for i in range(len(imglist)):
	imgname = imglist[i]
	img = Image.open(direction + "/" + imgname)
	img = img.convert("L")
	img = img.resize((128,128))
	img.save(imgname)






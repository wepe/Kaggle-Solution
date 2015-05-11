#*-*coding:utf8*-*#

import os

from PIL import Image
import numpy as np

#去除原图像左右上下的黑边
#按列计算sum，根据sum来确定黑边，如下所示：min+(max-min)/50是多次尝试后确定的一个不错的阈值
def crop(img):
	imgarray = np.asarray(img,dtype="uint8")  #数据类型设置为uint8，这种类型才能生成图像
	arr0 = imgarray[:,:,0]+imgarray[:,:,1]+imgarray[:,:,2]
	#去除左右黑边
	sum_column = arr0.sum(axis=0)
	i,j = 0,len(sum_column)-1
	start_column,end_column = i,j
	while(sum_column[i] < sum_column.min()+(sum_column.max()-sum_column.min())/50):
		start_column = i
		i += 1
	while(sum_column[j]< sum_column.min()+(sum_column.max()-sum_column.min())/50):
		end_column = j
		j -= 1
	#去除上下黑边
	sum_row = arr0.sum(axis=1)
	i,j = 0,len(sum_row)-1
	start_row,end_row = i,j
	while(sum_row[i]< sum_row.min()+(sum_row.max()-sum_row.min())/50):
		start_row = i
		i += 1
	while(sum_row[j]< sum_row.min()+(sum_row.max()-sum_row.min())/50):
		end_row = j
		j -= 1
	#截取保存,end_column-start_column是宽度，end_row-start_row是高度
	newarray = imgarray[start_row:end_row,start_column:end_column,:]
	img = Image.fromarray(newarray,"RGB")
	return img

#输入的img大小是64*height，整成64*64
#width=64,height可能大于64可能小于64
def tosquare(img):
	width,height = img.size   
	if height<width:
		black_len = (width - height)/2
		imgarray = np.asarray(img)
		newarray = np.zeros((64,64,3),dtype="uint8")
		newarray[black_len:black_len+height,:,:]=imgarray[:,:,:]
		img = Image.fromarray(newarray,"RGB")
	if height>width:
		l = height - width
		imgarray = np.asarray(img)
		img = Image.fromarray(imgarray[0:height-l,:,:],"RGB")
	return img


#"./train"原始训练数据，对测试数据"./test"做同样的预处理
if __name__ == "__main__":
	direction = "./train"
	save_path = "./train_RGB64"
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	imglist = os.listdir(direction)
	for i in xrange(len(imglist)):
		imgname = imglist[i]
		print imgname
		img = Image.open(direction+"/"+imgname)      
		img = crop(img)				     #截取图像，去除黑边
		width,height = img.size
		img = img.resize((64,64*height/width))     #resize image,保留宽高比
		img = tosquare(img)			     #整成正方形64*64*3
		img.save(save_path+"/"+imgname)
	

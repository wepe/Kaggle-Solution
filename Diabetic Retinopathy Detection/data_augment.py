"""
data augment:
	train_RGB64 -> train_RGB64_plus

"""

import os,shutil
import cv2
import csv


def get_trainlabel():
	table = {}
	rows = csv.reader(open( './trainLabels.csv','rb'))
	rows.next()
	for row in rows:
		table[row[0]] = int(row[1])
	return table

table = get_trainlabel()
srcdir = './train_RGB64/'
newdir = './trian_RGB64_plus/'
os.mkdir(newdir)	
imgs = os.listdir(srcdir)
(rows,cols) = (64,64)
category_0 = 25810/2
for imgname in imgs:
	label = table[imgname.split('.')[0]]
	if label == 0 and category_0>0:
		img = cv2.imread(srcdir+imgname)
		cv2.imwrite(newdir+imgname,img)
		category_0 -= 1
	elif label == 1:
		img = cv2.imread(srcdir+imgname)
		M1 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		M2 = cv2.getRotationMatrix2D((rows/2,cols/2),-3,1)
		M3 = cv2.getRotationMatrix2D((rows/2,cols/2),6,1)
		M4 = cv2.getRotationMatrix2D((rows/2,cols/2),-6,1)
		img1 = cv2.warpAffine(img,M1,(cols,rows))
		img2 = cv2.warpAffine(img,M2,(cols,rows))
		img3 = cv2.warpAffine(img,M3,(cols,rows))
		img4 = cv2.warpAffine(img,M4,(cols,rows))
		cv2.imwrite(newdir+imgname.split('.')[0]+'.0.jpg',img)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.1.jpg',img1)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.2.jpg',img2)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.3.jpg',img3)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.4.jpg',img4)		
	elif label == 2:
		img = cv2.imread(srcdir+imgname)
		M1 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		img1 = cv2.warpAffine(img,M1,(cols,rows))
		cv2.imwrite(newdir+imgname.split('.')[0]+'.0.jpg',img)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.1.jpg',img1)
	elif label ==3:
		img = cv2.imread(srcdir+imgname)
		img0 = cv2.flip(img,1)
		M1 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		M2 = cv2.getRotationMatrix2D((rows/2,cols/2),-3,1)
		M3 = cv2.getRotationMatrix2D((rows/2,cols/2),6,1)
		M4 = cv2.getRotationMatrix2D((rows/2,cols/2),-6,1)
		M5 = cv2.getRotationMatrix2D((rows/2,cols/2),9,1)
		M6 = cv2.getRotationMatrix2D((rows/2,cols/2),-9,1)
		M7 = cv2.getRotationMatrix2D((rows/2,cols/2),9,1)
		M8 = cv2.getRotationMatrix2D((rows/2,cols/2),-9,1)
		M9 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		M10 = cv2.getRotationMatrix2D((rows/2,cols/2),-3,1)
		M11 = cv2.getRotationMatrix2D((rows/2,cols/2),6,1)
		M12 = cv2.getRotationMatrix2D((rows/2,cols/2),-6,1)
		img1 = cv2.warpAffine(img,M1,(cols,rows))
		img2 = cv2.warpAffine(img,M2,(cols,rows))
		img3 = cv2.warpAffine(img,M3,(cols,rows))
		img4 = cv2.warpAffine(img,M4,(cols,rows))
		img5 = cv2.warpAffine(img,M5,(cols,rows))
		img6 = cv2.warpAffine(img,M6,(cols,rows))
		img7 = cv2.warpAffine(img0,M7,(cols,rows))
		img8 = cv2.warpAffine(img0,M8,(cols,rows))
		img9 = cv2.warpAffine(img0,M9,(cols,rows))
		img10 = cv2.warpAffine(img0,M10,(cols,rows))
		img11 = cv2.warpAffine(img0,M11,(cols,rows))
		img12 = cv2.warpAffine(img0,M12,(cols,rows))
		cv2.imwrite(newdir+imgname.split('.')[0]+'.0.jpg',img)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.1.jpg',img0)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.2.jpg',img1)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.3.jpg',img2)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.4.jpg',img3)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.5.jpg',img4)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.6.jpg',img5)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.7.jpg',img6)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.8.jpg',img7)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.9.jpg',img8)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.10.jpg',img9)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.11.jpg',img10)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.12.jpg',img11)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.13.jpg',img12)	
	elif label == 4:
		img = cv2.imread(srcdir+imgname)
		img0 = cv2.flip(img,1)
		M1 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		M2 = cv2.getRotationMatrix2D((rows/2,cols/2),-3,1)
		M3 = cv2.getRotationMatrix2D((rows/2,cols/2),6,1)
		M4 = cv2.getRotationMatrix2D((rows/2,cols/2),-6,1)
		M5 = cv2.getRotationMatrix2D((rows/2,cols/2),9,1)
		M6 = cv2.getRotationMatrix2D((rows/2,cols/2),-9,1)
		M7 = cv2.getRotationMatrix2D((rows/2,cols/2),9,1)
		M8 = cv2.getRotationMatrix2D((rows/2,cols/2),-9,1)
		M9 = cv2.getRotationMatrix2D((rows/2,cols/2),3,1)
		M10 = cv2.getRotationMatrix2D((rows/2,cols/2),-3,1)
		M11 = cv2.getRotationMatrix2D((rows/2,cols/2),6,1)
		M12 = cv2.getRotationMatrix2D((rows/2,cols/2),-6,1)
		img1 = cv2.warpAffine(img,M1,(cols,rows))
		img2 = cv2.warpAffine(img,M2,(cols,rows))
		img3 = cv2.warpAffine(img,M3,(cols,rows))
		img4 = cv2.warpAffine(img,M4,(cols,rows))
		img5 = cv2.warpAffine(img,M5,(cols,rows))
		img6 = cv2.warpAffine(img,M6,(cols,rows))
		img7 = cv2.warpAffine(img0,M7,(cols,rows))
		img8 = cv2.warpAffine(img0,M8,(cols,rows))
		img9 = cv2.warpAffine(img0,M9,(cols,rows))
		img10 = cv2.warpAffine(img0,M10,(cols,rows))
		img11 = cv2.warpAffine(img0,M11,(cols,rows))
		img12 = cv2.warpAffine(img0,M12,(cols,rows))
		cv2.imwrite(newdir+imgname.split('.')[0]+'.0.jpg',img)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.1.jpg',img0)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.2.jpg',img1)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.3.jpg',img2)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.4.jpg',img3)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.5.jpg',img4)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.6.jpg',img5)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.7.jpg',img6)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.8.jpg',img7)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.9.jpg',img8)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.10.jpg',img9)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.11.jpg',img10)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.12.jpg',img11)
		cv2.imwrite(newdir+imgname.split('.')[0]+'.13.jpg',img12)

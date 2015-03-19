代码文件说明
-----
###1、依赖库/软件
- PIL
- skimage
- NumPy


###2、数据预处理
- preprocess.py   
	
	作用于原图像，去除上下左右多余的黑边，截取出眼球部分。最后保存为256*height的图像，width=256，height按原图像的尺寸比例确定。

- gray.py

	将图像转换为灰度图。

- lbp.py

	提取lbp特征。	
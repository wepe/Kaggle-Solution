代码文件说明
-----
###1、依赖库/软件
- PIL
- NumPy
- OpenCV(Python接口)
- Keras(深度学习框架)

###2、数据预处理
- preprocess.py   
	
	作用于原始图像，去除上下左右多余的黑边，截取出眼球部分。最终将所有图像处理成同样的大小64*64。运行`preprocess.py`，其处理效果如下图所示：

	处理前：![](http://i.imgur.com/1aQEfe1.jpg)   
	处理后：![](http://i.imgur.com/WyfF7JO.jpg)




###3、样本平衡

原始样本的类别分布不平衡：

	类别 数目  比例
	0 25810  0.7347
	1 2443   0.0695
	2 5292   0.1506
	3 873    0.0248
	4 708    0.0201

需要做样本平衡，方案：

	类别０：欠采样，25810/2=12905
	类别１：旋转4次加上原图　2443*5=12215
	类别２：旋转1次加上原图  5292*2=10584
	类别３：水平翻转，每张图左右各旋转３次，873*2*7=12222
	类别４：水平翻转，每张图左右各旋转３次，708*2*7=9912

运行 `data_augment.py`, 生成的图像的效果：

![](http://i.imgur.com/u7s6005.jpg)![](http://i.imgur.com/4ouuT7x.jpg)![](http://i.imgur.com/ySqgatR.jpg)![](http://i.imgur.com/mhetuoN.jpg)![](http://i.imgur.com/wrZ3Kk6.jpg)


###4、测试样本类别分布估计

根据训练样本的分布比例，估计测试样本(53576个)的类别分布：

	0  39362
	1  3723
	2  8068
	3  1328
	4  1076

###5、构建训练CNN

- `data.py` 定义了加载训练数据的函数，供 `cnn.py`调用
- `cnn.py` 构建CNN,其结构如下（基于keras框架）：

		model = Sequential()
		model.add(Convolution2D(4, 3, 7, 7, border_mode='valid')) 
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		   
		model.add(Convolution2D(8,4, 5, 5, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		
		model.add(Convolution2D(16, 8, 3, 3, border_mode='valid')) 
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		
		model.add(Flatten())
		model.add(Dense(16*5*5, 256, init='normal'))
		model.add(Activation('relu'))
		
		model.add(Dense(256, 128, init='normal'))
		model.add(Activation('relu'))
			
		model.add(Dense(128, nb_classes, init='normal'))
		model.add(Activation('softmax'))

运行`cnn.py`，得到线下accuracy为71.％， 线上得分为0.1多。看一下预测的类别分布情况：

	0 31016
	1 4251
	2 14106
	3 2353
	4 1850


###6、To Do

- 64x64的分辨率是否影响到了识别率？ 试试 256*256
- CNN用更深/更浅的网络，卷积核数目/大小
- 不做分类，用回归做
- CNN as feature extractor，train SVM
- 训练多个模型


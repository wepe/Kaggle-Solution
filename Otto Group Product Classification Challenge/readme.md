代码文件说明
-----
###1、依赖库/软件
- sklearn
- NumPy


###2、数据预处理
- preprocess.py   
	
	数据预处理。包括加载训练数据集、测试数据集、归一化、零均值化。此外还定义了生成submission.csv的函数。



###3、分类

- KNN

	K近邻算法，效果很差。又费时又费内存。没调过参数。『放弃』

- RandomForest.py

	随机森林，调参中。beat the benchmark

- ExtraTrees.py

	未调参。

- GradientBoosting.py

	未调参。

- Adaboost.py

	未调参。
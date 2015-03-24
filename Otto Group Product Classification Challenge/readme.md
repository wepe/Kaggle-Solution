代码文件说明
-----
###1、依赖库/软件
- sklearn
- numpy


###2、数据预处理
- preprocess.py   
	
	一些预处理工作。定义了以下通用函数：加载训练数据集的函数loadTrainSet()、加载测试数据集的函数loadTestSet()、评估模型logloss值的函数evaluation()、生成提交文件submission.csv的函数saveResult()。

	数据经过归一化、零均值化。



###3、分类

- KNN

	K近邻算法，效果较差。费时又费内存。k=20时，logloss约为1。『放弃』

- RandomForest.py

	随机森林，n_estimators=400时，logloss约为0.55。

- ExtraTrees.py

	未调参。

- GradientBoosting.py

	未调参。

- Adaboost.py

	未调参。




代码使用
--
从官网下载train.csv、test.csv，与preprocess.py、RandomForest.py放在同一个目录下，直接运行RandomForest.py即可。(其他分类文件同理。)



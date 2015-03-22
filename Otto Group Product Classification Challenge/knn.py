import time
import preprocess

import sklearn


print "loading data..."
#load data in train.csv, divided into train data and validation data
data,label = preprocess.loadTrainSet()
val_data = data[0:5000]
val_label = label[0:5000]
train_data = data[5000:]
train_label = label[5000:]
#load data in test.csv
test_data = preprocess.loadTestSet()


print "start training KNN Classifier..."
#knn to classify.  k=20, score 7.79288.  
#It takes a lot of time to run the code,and occupy memory. Unfortunately，low accuracy.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
def knn(train_data,train_label,val_data,val_label,test_data):
	knnClf = KNeighborsClassifier(n_neighbors=20)
	knnClf.fit(train_data,train_label)
	#evaluate on validation set.
	score = knnClf.score(val_data,val_label)
	print "validation accuracy:",score
	#save the classifier
	#joblib.dump(knnClf, 'knnClf.pkl')

	#classify test set
	test_label = knnClf.predict(test_data)
	preprocess.saveResult(test_label,filename = "knn_submission.csv")



if __name__ == "__main__":
	t1 = time.time()
	knn(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print "Done! It cost",t2-t1,"s"

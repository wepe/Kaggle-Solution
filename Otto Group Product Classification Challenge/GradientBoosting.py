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

print "start training GradientBoosting..."
#Gradient Tree Boosting
#support "warm_start":When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble,
#otherwise, just erase the previous solution
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
def gb(train_data,train_label,val_data,val_label,test_data):
	gbClf = GradientBoostingClassifier()       # params: by default
	gbClf.fit(train_data,train_label)
	#evaluate on validation set.
	score = gbClf.score(val_data,val_label)
	print "validation accuracy:",score
	#save the classifier
	#joblib.dump(gbClf, 'gbClf.pkl')

	#classify test set
	test_label = gbClf.predict(test_data)
	preprocess.saveResult(test_label,filename = "GradientBoosting_submission.csv")
  

if __name__ == "__main__":
	t1 = time.time()
	gb(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print "Done! It cost",t2-t1,"s"
	
	

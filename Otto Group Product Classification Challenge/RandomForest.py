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


print "start training random forest..."
#random forest to classify.  
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
def rf(train_data,train_label,val_data,val_label,test_data):
	rfClf = RandomForestClassifier(n_estimators=75,criterion='gini',max_features=20,n_jobs=-1)
	rfClf.fit(train_data,train_label)
	#evaluate on validation set.
	score = rfClf.score(val_data,val_label)
	print "validation accuracy:",score
	#save the classifier
	#joblib.dump(rfClf, 'rfClf.pkl')

	#classify test set
	test_label = rfClf.predict(test_data)
	preprocess.saveResult(test_label,filename = "RandomForest_submission.csv")



if __name__ == "__main__":
	t1 = time.time()
	rf(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print "Done! It cost",t2-t1,"s"
	
	

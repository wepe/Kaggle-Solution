import cPickle,csv,os
import numpy as np
from PIL import Image

def load_test_data():
	test_data = np.empty((53576,3,64,64),dtype="float32")
	imgs = os.listdir("./test_RGB64")
	num = len(imgs)
	names = []
	for i in range(num):
		imgname = imgs[i]
		img = Image.open("./test_RGB64/"+imgname)
		arr = np.asarray(img,dtype="float32")
		test_data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		names.append(imgname.split('.')[0])
	return test_data,names

def model_predict():
	test_data,names = load_test_data()
	model = cPickle.load(open('model.pkl','rb'))
	test_label = model.predict_classes(test_data,batch_size=1, verbose=1)
	
	#create submission file
	w = csv.writer(open('submission.csv','wb'))
	w.writerow(('image','level'))
	length = len(test_label)
	table = {0:0,1:0,2:0,3:0,4:0}
	#prediction's distribution
	for i in range(length):
		w.writerow((names[i],test_label[i]))
		table[test_label[i]] += 1
	for key in table:
		print key,table[key]
	
if __name__ == "__main__":
	model_predict()

	
	


#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_color_img
import random,cPickle


nb_epoch = 200
batch_size = 200
nb_classes = 5

data, label0 = load_color_img("./trian_RGB64_plus/")
num = len(label0)
random.seed(12345)
index = [i for i in range(num)]
random.shuffle(index)
data = data[index]
label0 = label0[index]


label = np_utils.to_categorical(label0, nb_classes)


def create_model():
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
	return model


model = create_model()
sgd = SGD(l2=0.001,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

(X_train,X_val) = (data[0:50000],data[50000:])
(Y_train,Y_val) = (label[0:50000],label[50000:])
nb_train = len(Y_train)
nb_validation = len(Y_val)
print( 'train samples:',nb_train, 'validation samples:',nb_validation)
    

best_accuracy = 0.0
for e in range(nb_epoch):
    print('Epoch', e)
    print("Training...")
    batch_num = len(Y_train)/batch_size
    progbar = generic_utils.Progbar(X_train.shape[0])
    for i in range(batch_num):
        train_loss,train_accuracy = model.train(X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size],accuracy=True)
        progbar.add(batch_size, values=[("train loss", train_loss),("train accuracy:", train_accuracy)] )
        
    print("Validation...")
    val_loss,val_accuracy = model.evaluate(X_val, Y_val, batch_size=1,show_accuracy=True)
    if best_accuracy<val_accuracy:
        best_accuracy = val_accuracy
        cPickle.dump(model,open("./model.pkl","wb"))



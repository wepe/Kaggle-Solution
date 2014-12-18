# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 22:46:03 2014

@author: wepon

@blog:http://blog.csdn.net/u012162613
"""

from numpy import *
import csv

def loadData(csvName):
    l=[]
    file=open(csvName)
    lines=csv.reader(file)
    for line in lines:
        l.append(line) 
    l=array(l)   
    return toInt(l)

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=float(array[i,j])
    return newArray

#result是结果列表 
#csvName是存放结果的csv文件名，saveResult将result存储成列形式
def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
            
#调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier  
def knnClassify(trainData,trainLabel,testData): 
    knnClf=KNeighborsClassifier(10)#default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData,ravel(trainLabel))
    testLabel=knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel
    
#调用logistics回归算法
from sklearn.linear_model import LogisticRegression
def logisticRegressionClf(trainData,trainLabel,testData):
    lrClf=LogisticRegression()
    lrClf.fit(trainData,ravel(trainLabel))
    testLabel=lrClf.predict(testData)
    saveResult(testLabel,'sklearn_lr_Result.csv')
    return testLabel
    
#调用scikit的SVM算法包
from sklearn import svm   
def svcClassify(trainData,trainLabel,testData): 
    svcClf=svm.SVC(C=5) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
    svcClf.fit(trainData,ravel(trainLabel))
    testLabel=svcClf.predict(testData)
    saveResult(testLabel,'sklearn_SVC_C=5.0_Result.csv')
    return testLabel

#调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
def GaussianNBClassify(trainData,trainLabel,testData): 
    nbClf=GaussianNB()          
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'sklearn_GaussianNB_Result.csv')
    return testLabel
    
from sklearn.naive_bayes import MultinomialNB   #nb for 多项式分布的数据    
def MultinomialNBClassify(trainData,trainLabel,testData): 
    nbClf=MultinomialNB(alpha=0.1)      #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.       
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return testLabel
    
def DataScienceLondon():
    trainData=loadData('train.csv')
    trainLabel=loadData('trainLabels.csv')
    testData=loadData('test.csv')
    #使用不同算法
    knnClassify(trainData,trainLabel,testData)     #
    svcClassify(trainData,trainLabel,testData)     #
    logisticRegressionClf(trainData,trainLabel,testData)
    GaussianNBClassify(trainData,trainLabel,testData)  #效果最好
#    MultinomialNBClassify(trainData,trainLabel,testData)
#MultinomialNBClassify会报错ValueError: Input X must be non-negative
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 21:59:00 2014

@author: wepon

@blog:http://blog.csdn.net/u012162613
"""

from numpy import *
import csv

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array
    
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784
    #return trainData,trainLabel
    
def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))  #  data 28000*784
    #return testData
    
def loadTestResult():
    l=[]
    with open('knn_benchmark.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])  #  label 28000*1
    
#result是结果列表 
#csvName是存放结果的csv文件名
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
    knnClf=KNeighborsClassifier()#default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData,ravel(trainLabel))
    testLabel=knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel
    
#调用scikit的SVM算法包
from sklearn import svm   
def svcClassify(trainData,trainLabel,testData): 
    svcClf=svm.SVC(C=5.0) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
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


def digitRecognition():
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    #使用不同算法
    result1=knnClassify(trainData,trainLabel,testData)
    result2=svcClassify(trainData,trainLabel,testData)
    result3=GaussianNBClassify(trainData,trainLabel,testData)
    result4=MultinomialNBClassify(trainData,trainLabel,testData)
    
    #将结果与跟给定的knn_benchmark对比,以result1为例
    resultGiven=loadTestResult()
    m,n=shape(testData)
    different=0      #result1中与benchmark不同的label个数，初始化为0
    for i in xrange(m):
        if result1[i]!=resultGiven[0,i]:
            different+=1
    print different
            

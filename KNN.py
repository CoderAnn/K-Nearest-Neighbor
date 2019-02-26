# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:05:13 2019

@author: Beauty
"""
from numpy import *
import os
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistaces=sqDiffMat.sum(axis=1)
    diatances=sqDistaces**0.5
    
    sortedDistIndicies=diatances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)#打开当前路径下名为filename的文件
    arrayOLines = fr.readlines()#返回当前文件的行数
    numberOfLine = len(arrayOLines)
    returnMat = zeros((numberOfLine,3))#因为这里具体的特征是3个
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(round(float(listFromLine[-1])))
        index += 1
    return returnMat,classLabelVector

def img2vector(filename):
    returnVect = zeros((1,1024)) #将图像转化为1*1024的向量
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % filenameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUndertest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUndertest,trainingMat,hwLabels,3)
        print("分类结果为: %d,真实的结果是：%d" % (classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount +=1.0
    print("\n判断错误个数是： %d" % errorCount)
    print("\n错误率是： %f" % (errorCount/float(mTest)))
        

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfileResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                        datingLabels[numTestVecs:m],3)
        print("分类器的结果是：%d,真实的情况是: %d" % (classfileResult,datingLabels[i]))
        if classfileResult != datingLabels[i]:
            errorCount += 1.0
    print("全部的错误率是：%f" % (errorCount/float(numTestVecs)))
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video game?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResults = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("你对这个人的喜欢程度大概是：", resultList[classifierResults - 1])
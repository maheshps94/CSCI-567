import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
import operator
from sklearn.datasets import load_boston
import itertools as it
from svmutil import *
from timeit import default_timer

def getData(fileName):
	mat = sp.io.loadmat(fileName)
		
	dataFrame= pd.DataFrame(mat['features'])
	labelValues = mat["label"]
	transformValues = [1, 6, 7, 13, 14, 25, 28]
	totalValues = list(range(0,30))
	remainingValues = list(set(totalValues) - set(transformValues))
	for i in range(len(transformValues)):
		listValues = list(dataFrame[transformValues[i]])
		listA = []
		listB = []
		listC = []
		for j in range(len(listValues)):
			if listValues[j] == -1:
				listA.append(1)
				listB.append(0)
				listC.append(0)
			elif listValues[j] == 0:
				listA.append(0)
				listB.append(1)
				listC.append(0)
			else:
				listA.append(0)
				listB.append(0)
				listC.append(1)
		dataFrame['col' + str(transformValues[i]) + '-1'] = pd.DataFrame(listA)
		dataFrame['col' + str(transformValues[i]) + '0'] = pd.DataFrame(listB)
		dataFrame['col' + str(transformValues[i]) + '1'] = pd.DataFrame(listC)
		dataFrame = dataFrame.drop(transformValues[i],1)
	for i in range(len(remainingValues)):
		listValues = list(dataFrame[remainingValues[i]])
		for j in range(len(listValues)):
			if listValues[j] == -1:
				listValues[j] = 0
		dataFrame[remainingValues[i]] = pd.DataFrame(listValues)
		
	return dataFrame, labelValues

def linearSVM(dataFrameTrain,labelValues):
	print "\n-----------------------------------------------------------------------------\n"	
	print "Linear SVM"
	dataList = dataFrameTrain.values.tolist()
	cList=[]
	x = -6
	startTime = default_timer()
	for i in range(0,9):
		cList.append(4**x)
		x +=1
	prob = svm_problem(labelValues.transpose(), dataList)
	for i in range(0,9):
		param = svm_parameter('-t 0 -v 3 -q -c ' + str(cList[i]))
		print "\nc = " + str(cList[i])
		m = svm_train(prob, param)
	print "The average training time for Linear SVM is " + str((default_timer() - startTime)/27.0) + " seconds"

def polynomialKernel(dataFrameTrain,labelValues):
	print "\n-----------------------------------------------------------------------------\n"
	print "Polynomial Kernel"
	maxVal = 0
	dataList = dataFrameTrain.values.tolist()
	cList=[]
	x = -3
	
	for i in range(0,11):
		cList.append(4**x)
		x +=1
	startTime = default_timer()
	prob = svm_problem(labelValues.transpose(), dataList)
	for degree in range(0,3):
		for i in range(0,11):
			param = svm_parameter('-t 1 -v 3 -q -c ' + str(cList[i]) + ' -d ' + str(degree + 1))
			print "\nc = " + str(cList[i]) + ", degree = " + str(degree+1)
			m = svm_train(prob, param) 
			if m > maxVal:
				maxVal = m
				c = cList[i]
				d = degree + 1
	print "Polynomial Kernel MAX is " + str(maxVal) + " at c 4^" + str(math.log(c,4)) + " and degree " + str(d)
	print "The average training time for Polynomial Kernel is " + str((default_timer() - startTime)/99.0) + " seconds"
	return maxVal,c,d

def rbfKernel(dataFrameTrain,labelValues):
	print "\n-----------------------------------------------------------------------------\n"
	print "RBF Kernel"
	maxVal =0
	dataList = dataFrameTrain.values.tolist()
	cList=[]
	gList = []
	x = -3
	for i in range(0,11):
		cList.append(4**x)
		x +=1
	y = -7
	for i in range(0,7):
		gList.append(4**y)
		y +=1
	startTime = default_timer()
	prob = svm_problem(labelValues.transpose(), dataList)
	for gamma in range(0,7):
		for i in range(0,11):
			param = svm_parameter('-t 2 -v 3 -q -c ' + str(cList[i]) + ' -g ' + str(gList[gamma]))
			print "\nc = " + str(cList[i]) + ", gamma = " + str(gList[gamma])
			m = svm_train(prob, param) 
			if m > maxVal:
				maxVal = m
				c = cList[i]
				g = gList[gamma]
	print "RBF Kernel MAX is " + str(maxVal) + " at c 4^" + str(math.log(c,4)) + " and gamma 4^" + str(math.log(g,4))
	print "The average training time for Polynomial Kernel is " + str((default_timer() - startTime)/231.0) + " seconds"
	return maxVal,c,g

def main():
	data = {}
	dataFrameTrain,trainLabel = getData('phishing-train.mat')
	linearSVM(dataFrameTrain,trainLabel)
	pval,pc,pld = polynomialKernel(dataFrameTrain,trainLabel)
	rval,rc,rg = rbfKernel(dataFrameTrain,trainLabel)
	dataFrameTest, testLabel =getData('phishing-test.mat')
	file = open('data.txt','w')
	data['polyValue'] = str(pval)
	data['rbfValue'] = str(rval)
	if pval > rval:
		c = pc
		d = pld
		data['C'] = str(c)
		data['val'] = str(d)
	else:
		c = rc
		g = rg
		data['c'] = str(c)
		data['val'] = str(g)
		file.write(str(data))
		file.close()

if __name__ == '__main__':
	import sys
	main()
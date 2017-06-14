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
		#print dataFrameTrain[transformValues[i]]
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
	#print remainingValues
	#print dataFrameTrain
	for i in range(len(remainingValues)):
		listValues = list(dataFrame[remainingValues[i]])
		for j in range(len(listValues)):
			if listValues[j] == -1:
				listValues[j] = 0
		dataFrame[remainingValues[i]] = pd.DataFrame(listValues)
		#print dataFrameTrain
	
	return dataFrame, labelValues

def predictPolynomial(dataFrameTrain,trainLabel,dataFrameTest,testLabel,c,d):
	dataList = dataFrameTrain.values.tolist()
	dataTestList = dataFrameTest.values.tolist()
	prob = svm_problem(trainLabel.transpose(), dataList)
	param = svm_parameter('-t 1 -q -c ' + str(c) + ' -d ' + str(d))
	m = svm_train(prob, param) 
	label, accuracy, value = svm_predict(testLabel.transpose(),dataTestList,m)
	#print "Test Accuracy " + str(accuracy)

def predictrbf(dataFrameTrain,trainLabel,dataFrameTest,testLabel,c,g):
	dataList = dataFrameTrain.values.tolist()
	dataTestList = dataFrameTest.values.tolist()
	prob = svm_problem(trainLabel.transpose(), dataList)
	param = svm_parameter('-t 2 -q -c ' + str(c) + ' -g ' + str(g))
	m = svm_train(prob, param) 
	label, accuracy, value = svm_predict(testLabel.transpose(),dataTestList,m)

def main():
	print "\n-----------------------------------------------------------------------------\n"
	dataFrameTrain,trainLabel = getData('phishing-train.mat')
	dataFrameTest, testLabel =getData('phishing-test.mat')
	try:
		file = open('data.txt','r').read()
		data = eval(file)
		c = data['c']
		polyValue = data['polyValue']
		rbfValue = data['rbfValue']
		val = data['val']
		if polyValue > rbfValue:
			print "The selected kernel is Polynomial Kernel"
			print "Training Accuracy "  + str(polyValue)
			print "Value of C " + str(c)
			print "Value of degree " + str(val)
			print "\n-----------------------------------------------------------------------------\n"
			print " Prediction using Polynomial Kernel\n"
			predictPolynomial(dataFrameTrain,trainLabel,dataFrameTest,testLabel,c,val)
		else:
			print "The selected kernel is RBF Kernel"
			print "Training Accuracy "  + str(polyValue)
			print "Value of C " + str(c)
			print "Value of gamma " + str(val)
			print "\n-----------------------------------------------------------------------------\n"
			print "Prediction using RBF Kernel\n"
			predictrbf(dataFrameTrain,trainLabel,dataFrameTest,testLabel,c,val)
	except IOError:
		print "Please run lkSVM.py or CSCI567_hw3_fall16.py to generate input file containing kernel function details"

if __name__ == '__main__':
	import sys
	main()
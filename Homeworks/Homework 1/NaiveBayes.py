import pandas as pd
import numpy as np
import operator
import math
from scipy.stats import norm
from collections import Counter
def getData():
	global dataFrame,arrTrain,column,meanDataFrame,stdDataFrame,prior
	dataFrame = pd.read_csv('train.txt', header =None)
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	dataFrame.columns =column
	arrTrain = list(dataFrame.values.tolist())
	mu = {}
	sigma = {}
	meanDataFrame = dataFrame.groupby(['Class']).mean().reset_index()
	stdDataFrame = dataFrame.groupby(['Class']).std().reset_index()
	prior = dataFrame.groupby(['Class']).count().drop(['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'],1).reset_index()/196
	
def getNB(arrTest,mean,std,priorvalue):
	likelihood = 0
	for y in column[1:-1]:
		if std[y] == 0:
			likelihood += 0
		else:
			likelihood +=  norm.logpdf(arrTest[y],mean[y],std[y]) #np.log((1.0/(math.sqrt(2 * math.pi) * std[y])) * math.exp(-1.0 / (2 * ((std[y])**2))) * ((arrTest[y] - mean[y]) ** 2))
	return priorvalue + likelihood

def getAccuracy(fileType):
	if fileType == 0:
		dataFrameTest = pd.read_csv('test.txt', header =None)
	else:
		dataFrameTest = pd.read_csv('train.txt', header =None)
	dataFrameTest.columns =column
	nbvalues = [[0]*len(meanDataFrame) for i in range(len(dataFrameTest))]
	for i in range(len(dataFrameTest)):
		for j in range(len(meanDataFrame)):
			nbvalues[i][j] =  getNB(dataFrameTest.loc[i],meanDataFrame.loc[j],stdDataFrame.loc[j],np.log(prior.loc[j]['Id']))
	classPrediction = []
	for i in range(len(dataFrameTest)):
		index, value = max(enumerate(nbvalues[i]), key=operator.itemgetter(1))
		index = index +1
		if index >=4 :
			index = index + 1
		classPrediction.append(index)
	acc =  [i for i, j in zip(classPrediction, list(dataFrameTest["Class"])) if i == j]
	print str(round(float(len(acc)) / float(len(list(dataFrameTest["Class"]))) * 100,2)) + "%"

def main():
	getData() 
	print "\nTesting Accuracy for Naive Bayes"
	getAccuracy(0)
	print "\nTraining Accuracy for Naive Bayes"
	getAccuracy(1)
if __name__ == '__main__':
	import sys
	main()
	

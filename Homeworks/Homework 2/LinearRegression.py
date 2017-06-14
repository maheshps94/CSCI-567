import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
import operator
from sklearn.datasets import load_boston
import itertools as it

def getData():
	global column, targetValuesTrain,targetValuesTest
	boston = load_boston()
	keys = boston.keys()
	dataFrameTrain= pd.DataFrame(boston.data)
	column = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
	dataFrameTrain.columns = column
	dataFrameTrain['MEDV'] = boston.target
	dataFrameTest =  dataFrameTrain.loc[lambda dataFrameTrain: dataFrameTrain.index%7 == 0]
	dataFrameTest = (dataFrameTest.reset_index()).drop(['index'],1)
	dataFrameTrain = dataFrameTrain.loc[lambda dataFrameTrain: dataFrameTrain.index%7 != 0]
	dataFrameTrain = (dataFrameTrain.reset_index()).drop(['index'],1)
	return dataFrameTrain,dataFrameTest	

def getHistogram(dataFrameTrain):
	print "\n-----------------------------------------------------------------------------\n"
	print "Plotting the Histograms"	
	print "Close all the histogram plots to continue with the program execution"
	for c in column:
		dataFrameTrain.hist(column=c,bins =10)
	plot.show()

	print "\n-----------------------------------------------------------------------------\n"

def getPearsonValues(dataFrameTrain,dataFrameTest,targetColumn,col):
	global pearsonValues
	targetColumn = np.array(targetColumn)
	droppedCol = None
	newColumn = list(column)
	if col[-1] in dataFrameTrain:
		dataFrameTrain.drop(col,1)
		if col[-1] in newColumn:
			newColumn.remove(col[-1])
	pearsonValue = {}
	pearsonVal = {}
	junkValue = []
	
	for c in newColumn:
		pV, jV = sp.stats.pearsonr(dataFrameTrain[c],targetColumn)
		pearsonValue[column.index(c)] = abs(pV)
		pearsonVal[column.index(c)] = pV
		junkValue.append(jV)
	pearsonSorted = sorted(pearsonValue.items(), key = operator.itemgetter(1), reverse = True)
	pearsonValues = pearsonVal.items()
	return pearsonSorted

def featureSelection(part,pearsonSorted,dataFrameTrain,dataFrameTest):
	
	c = list(column)
	pearsonColumn = []
	if part == 0:
		for i in range(4):
			pearsonColumn.append(column[pearsonSorted[i][0]])	
			c.remove(column[pearsonSorted[i][0]])
		
		dataFrameTrain = dataFrameTrain.drop(c,1)
		dataFrameTest = dataFrameTest.drop(c,1)
		print pearsonColumn
		return dataFrameTrain, dataFrameTest, c
	else:
		for val in range(len(pearsonSorted)):
			if column[pearsonSorted[val][0]] in cols:
				continue
			else:
				cols.append(column[pearsonSorted[val][0]])
				break
		for k in cols:
			c.remove(k)
		
		dataFrameTrain = dataFrameTrain.drop(c,1)
		dataFrameTest = dataFrameTest.drop(c,1)

		return dataFrameTrain, dataFrameTest, cols

def bruteForceSelection(dataFrameTrain,dataFrameTest):
	combinationColumns = list(it.combinations(column,4))
	bruteTestError = []
	bruteTrainError = []
	for i in range(len(combinationColumns)):
		retainColumns = []
		for j in range(len(list(combinationColumns[i]))):
			if combinationColumns[i][j] in column:
				retainColumns.append(combinationColumns[i][j])
		dropColumn = list(column)
		for k in range(len(retainColumns)):
			dropColumn.remove(retainColumns[k])
		bruteDataFrameTrain = dataFrameTrain.drop(dropColumn,1)
		x, y, testError, trainError = linearRegression(bruteDataFrameTrain,bruteDataFrameTrain)
		bruteTestError.append(testError)
		bruteTrainError.append(trainError)
	index = bruteTestError.index(min(bruteTrainError))
	print list(combinationColumns[index])
	temp = list(combinationColumns[index])
	dropColumn = list(column)
	for k in range(len(temp)):
			dropColumn.remove(temp[k])
	bruteDataFrameTrain = dataFrameTrain.drop(dropColumn,1)
	bruteDataFrameTest = dataFrameTest.drop(dropColumn,1)	
	x, y, testError, trainError = linearRegression(bruteDataFrameTrain,bruteDataFrameTest)
	print "Train Mean Square Error = " + str(trainError)
	print "Test Mean Square Error = " + str(testError)

def featureExpansion(dataFrameTrain,dataFrameTest):
	meanFrameTrain = dataFrameTrain.mean()
	stdFrameTrain = dataFrameTrain.std()
	normalFrameTrain = (dataFrameTrain - meanFrameTrain)/stdFrameTrain
	normalFrameTest = (dataFrameTest - meanFrameTrain)/stdFrameTrain
	for i in range(len(column)):
		for j in range(i,len(column)):
			dataFrameTrain[str(column[i] + "*" + str(column[j]))] = normalFrameTrain[column[i]] * normalFrameTrain[column[j]]
			dataFrameTest[str(column[i] + "*" + str(column[j]))] = normalFrameTest[column[i]] * normalFrameTest[column[j]]
	return dataFrameTrain,dataFrameTest

def linearRegression(dataFrameTrain,dataFrameTest):
	meanFrameTrain = dataFrameTrain.mean()
	stdFrameTrain = dataFrameTrain.std()
	normalFrameTrain = (dataFrameTrain - meanFrameTrain)/stdFrameTrain
	normalFrameTest = (dataFrameTest - meanFrameTrain)/stdFrameTrain
	normalFrameTrain.insert(0,'ONES',1)
	normalFrameTest.insert(0,'ONES',1)
	normalFrameTrain = normalFrameTrain.drop(['MEDV'],1)
	normalFrameTest = normalFrameTest.drop(['MEDV'],1)
	targetValuesTrain = np.array(dataFrameTrain.MEDV)
	targetValuesTest = np.array(dataFrameTest.MEDV)
	numpyTrainMatrix = normalFrameTrain.as_matrix()
	numpyTestMatrix = normalFrameTest.as_matrix()
	lmsTrainMatrix = np.dot(np.dot(np.linalg.pinv(np.dot(numpyTrainMatrix.transpose(),numpyTrainMatrix)),numpyTrainMatrix.transpose()), targetValuesTrain)
	predictionValuesTrain = []
	predictionValuesTest = []
	for i in range(len(numpyTestMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTestMatrix[i])
		predictionValuesTest.append(d)
	for i in range(len(numpyTrainMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTrainMatrix[i])
		predictionValuesTrain.append(d)
	colDifferenceTest = predictionValuesTest - targetValuesTest
	testMeanSquareError = (colDifferenceTest ** 2).mean()
	colDifferenceTrain = predictionValuesTrain - targetValuesTrain
	trainMeanSquareError = (colDifferenceTrain ** 2).mean()
	return colDifferenceTrain, colDifferenceTest,testMeanSquareError,trainMeanSquareError

def secondMain(dataFrameTrain,dataFrameTest,pearsonSorted):
	print "Linear Regression for Features with four of the highest Pearson Co-relation Co-efficients Selection\n"
	dataFrameTrainFeature,dataFrameTestFeature, col = featureSelection(0,pearsonSorted,dataFrameTrain,dataFrameTest)
	colDifferenceTrain,colDifferenceTest,testError,trainError = linearRegression(dataFrameTrainFeature,dataFrameTestFeature)
	print "Train Mean Square Error = " + str(trainError)
	print "Test Mean Square Error = " + str(testError)
	print "\n-----------------------------------------------------------------------------\n"

	print "Linear Regression for Features with four of the highest Pearson Co-relation Co-efficients using Residue Values\n"
	for iter in range(4):
		if iter > 0:
			pearsonSorted = getPearsonValues(dataFrameTrain,dataFrameTest,colDifferenceTrain,cols)
		dataFrameTrainFeature,dataFrameTestFeature, col = featureSelection(1,pearsonSorted,dataFrameTrain,dataFrameTest)
		colDifferenceTrain,colDifferenceTest,testError,trainError = linearRegression(dataFrameTrainFeature,dataFrameTestFeature)
		
	print cols
	print "Train Mean Square Error = " + str(trainError)
	print "Test Mean Square Error = " + str(testError)
	print "\n-----------------------------------------------------------------------------\n"
	
	print "Linear Regression for Features using Brute Force Selection\n"
	bruteForceSelection(dataFrameTrain,dataFrameTest)
	
	print "\n-----------------------------------------------------------------------------\n"
	print "Linear Regression for Features with Feature Expansion\n"
	newTrainFrame, newTestFrame = featureExpansion(dataFrameTrain,dataFrameTest)
	x, y, testError, trainError = linearRegression(newTrainFrame,newTestFrame)
	print "Train Mean Square Error = " + str(trainError)
	print "Test Mean Square Error = " + str(testError)
	print "\n-----------------------------------------------------------------------------\n"

def main():
	global cols
	cols = []
	dataFrameTrain,dataFrameTest = getData()
	printFrame =pd.DataFrame()
	tempList =[]

	getHistogram(dataFrameTrain)
	
	pearsonSorted = getPearsonValues(dataFrameTrain,dataFrameTest,dataFrameTrain.MEDV,'MEDV')
	print "Pearson Values for the Train Data\n"
	for i in range(len(pearsonValues)):
		tempList.append([column[pearsonValues[i][0]],pearsonValues[i][1]])
	printFrame = pd.DataFrame(tempList)
	printFrame.columns = ['Column','Pearson Co-efficient']
	print printFrame.to_string(index=False)
	print "\n-----------------------------------------------------------------------------\n"

	print "Linear Regression on Test and Train Data\n"
	x,y,testError,trainError = linearRegression(dataFrameTrain,dataFrameTest)
	print "Train Mean Square Error = " + str(trainError)
	print "Test Mean Square Error = " + str(testError)
	print "\n-----------------------------------------------------------------------------\n"
	return dataFrameTrain,dataFrameTest,pearsonSorted
if __name__ == '__main__':
	import sys
	main()
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
import operator
from sklearn.datasets import load_boston

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

def normalizeData(dataFrameTrain,dataFrameTest):
	global normalFrameTrain,normalFrameTest
	meanFrameTrain = dataFrameTrain.mean()
	stdFrameTrain = dataFrameTrain.std()
	normalFrameTrain = (dataFrameTrain - meanFrameTrain)/stdFrameTrain
	normalFrameTest = (dataFrameTest - meanFrameTrain)/stdFrameTrain
	normalFrameTrain.insert(0,'ONES',1)
	normalFrameTest.insert(0,'ONES',1)
	normalFrameTrain = normalFrameTrain.drop(['MEDV'],1)
	normalFrameTest = normalFrameTest.drop(['MEDV'],1)
	
def dataCrossValidation(dataFrameTrain,orgFrameTrain,orgFrameTest):
	dataFrameRandom = dataFrameTrain.iloc[np.random.permutation(len(dataFrameTrain))]
	splitDataFrames = np.array_split(dataFrameRandom, 10)
	testErrorList = [0]*6
	trainErrorList = [0]*6
	for i in range(len(splitDataFrames)):
		dataFrameTrainNew = pd.DataFrame()
		for j in range(len(splitDataFrames)):
			if i == j:
				dataFrameTest = splitDataFrames[i]
			else:
				dataFrameTrainNew = dataFrameTrainNew.append(splitDataFrames[j])
		lambdaValue = 0.0001
		index =0
		
		while lambdaValue <= 10:
			testError, trainError = ridgeRegression(i+1,dataFrameTrainNew,dataFrameTest,lambdaValue)
			testErrorList[index] += testError
			trainErrorList[index] +=trainError
			lambdaValue = lambdaValue * 10
			index += 1
	ind = 0
	for h in range(len(trainErrorList)):
		print "lambda = " + str(0.0001 * (10 ** ind))
		print "Train Mean Square Error = " +  str(trainErrorList[h]/len(splitDataFrames))
		print "Test Mean Square Error = " +  str(testErrorList[h]/len(splitDataFrames)) + "\n"
		ind += 1	
	print "-----------------------------------------------------------------------------\n"

	print "Ridge Regression after Cross Validation for different lambda values\n"
	lambdaValue = 0.0001
		
	while lambdaValue <= 10:
		testError, trainError = ridgeRegression(i+1,orgFrameTrain,orgFrameTest,lambdaValue)
		print "lambda = " + str(lambdaValue)
		print "Train Mean Square Error = " +  str(trainError)
		print "Test Mean Square Error = " +  str(testError) + "\n"
		lambdaValue = lambdaValue * 10
	
	
def ridgeRegression(binNumber,dataFrameTrain,dataFrameTest,lambdaValue):
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
	identityMatrix = lambdaValue * np.identity(len(numpyTrainMatrix.transpose()))
	lmsTrainMatrix = np.dot(np.dot(np.linalg.pinv(np.dot(numpyTrainMatrix.transpose(),numpyTrainMatrix)+ identityMatrix),numpyTrainMatrix.transpose()), targetValuesTrain)
	predictionValuesTrain = []
	predictionValuesTest = []
	for i in range(len(numpyTestMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTestMatrix[i])
		predictionValuesTest.append(d)
	for i in range(len(numpyTrainMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTrainMatrix[i])
		predictionValuesTrain.append(d)
	testError =((predictionValuesTest - targetValuesTest) ** 2).mean()
	trainError = ((predictionValuesTrain - targetValuesTrain) ** 2).mean()
	if binNumber > 0:
		return testError, trainError
	else:
		print "lambda = " + str(lambdaValue)
		print "Train Mean Square Error  " + str(trainError) 
		print "Test Mean Square Error  " + str(testError) + "\n"


def main():
	dataFrameTrain,dataFrameTest = getData()

	print "Ridge Regression\n"
	ridgeRegression(0,dataFrameTrain,dataFrameTest,0.01)
	ridgeRegression(0,dataFrameTrain,dataFrameTest,0.1)
	ridgeRegression(0,dataFrameTrain,dataFrameTest,1)
	print "-----------------------------------------------------------------------------\n"

	print "Ridge Regression with Cross Validation\n"
	dataCrossValidation(dataFrameTrain,dataFrameTrain,dataFrameTest)
	print "-----------------------------------------------------------------------------\n"

if __name__ == '__main__':
	import sys
	main()
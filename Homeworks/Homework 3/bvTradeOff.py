import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
from scipy.stats import norm
import operator
from sklearn.datasets import load_boston
import itertools as it
import random as rd 


def biasVarianceRidge(N,lambdaValue):
	dataFrameTrain= pd.DataFrame()
	dataFrameError = pd.DataFrame()	
	dataFrameTemp = pd.DataFrame()
	squareError =0
	g1List = []
	g2List = []
	g3List = []
	g4List = []
	g5List = []
	for i in range(100):
		errorList = []
		dataFrameTemp = pd.DataFrame()
		dataFrameTrain[i] = np.random.uniform(-1,1,N)
		ylist=[]
		
		for x in range(0,len(dataFrameTrain)):
			ylist.append(2*(dataFrameTrain.iloc[x][i]**2) + np.random.normal(0,math.sqrt(0.1)))
		dataFrameTrain['y'] = ylist
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp['y'] = dataFrameTrain['y']
		dataFrameTemp['y'] = 1 - dataFrameTemp['y']
		dataFrameTemp['y'] = dataFrameTemp['y'] ** 2
		errorList.append(dataFrameTemp['y'].mean())
		dataFrameTemp['y'] = dataFrameTrain['y']
		w, error = ridgeRegression(dataFrameTemp,lambdaValue)
		
		g1List.append(w)
		errorList.append(error)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp['y'] = dataFrameTrain['y']
		w1, error = ridgeRegression(dataFrameTemp,lambdaValue)
		
		g2List.append(w1)	
		errorList.append(error)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp['y'] = dataFrameTrain['y']
		dataFrameTemp["col" + str(2)] = dataFrameTemp[i]**2
		
		w2,error = ridgeRegression(dataFrameTemp,lambdaValue)
		errorList.append(error)
		g3List.append(w2)
		
		
		dataFrameTemp["col" + str(3)] = dataFrameTemp[i]**3
		w3,error = ridgeRegression(dataFrameTemp,lambdaValue)
		errorList.append(error)
		g4List.append(w3)
		
		dataFrameTemp["col" + str(4)] = dataFrameTemp[i]**4
		w4,error = ridgeRegression(dataFrameTemp,lambdaValue)
		errorList.append(error)
		g5List.append(w4)


		dataFrameError[i] = (errorList)
	g1 = np.sum(g1List,axis=0)/100
	g2 = np.sum(g2List,axis=0)/100
	g3 = np.sum(g3List,axis=0)/100
	g4 = np.sum(g4List,axis=0)/100
	g5 = np.sum(g5List,axis=0)/100
	
	biasI =0
	biasII = 0
	biasIII = 0
	biasIV = 0
	biasV = 0
	biasVI = 0
	
	varianceI = 0
	varianceII = 0
	varianceIII = 0
	varianceIV = 0
	varianceV = 0
	varianceVI = 0

	for i in range(100):
		errorList = []
		ylist =[]
		xlist =[]
		dataFrameTemp = pd.DataFrame()
		dataFrameTrain[i] = np.random.uniform(-1,1,N)
			
		for x in range(0,len(dataFrameTrain)):
			ylist.append(2*(dataFrameTrain.iloc[x][i]**2) + np.random.normal(0,math.sqrt(0.1)))
			xlist.append(dataFrameTrain.iloc[x][i])
		dataFrameTrain['y'] = ylist
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp = dataFrameTrain.copy(deep = True)
		#print dataFrameTrain['y']
		dataFrameTemp.insert(0,'ONES',1)
		dataFrameTemp['y'] = dataFrameTemp['ONES'] - dataFrameTemp['y']
		#print dataFrameTemp
		npTrain1 = np.array(dataFrameTemp['y'])
		#print npTrain1
		bias1 = (((npTrain1))**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		variance1 = 1 - dataFrameTemp['ONES'].mean()
		varianceI += variance1
		biasI += np.sum(bias1)
		
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp = dataFrameTrain.copy(deep=True)
		dataFrameTemp.insert(0,'ONES',1)
		dataFrameTemp['vals'] = dataFrameTemp['ONES'] * g1
		
		npTrain2 = np.array(dataFrameTemp['vals'])
		npVariance2 = np.array(dataFrameTemp['ONES'])
		
		bias2 = ((npTrain2 - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasII += np.sum(bias2)
		
		variance2 = ((g1List[i]* npVariance2) -  (npTrain2))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceII += np.sum(variance2)

		dataFrameTrain = dataFrameTrain.drop(['y'],1)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp.insert(0,'ONES',1)
		
		npTrain3 = np.array(dataFrameTemp)
		bias3 = ((np.dot(g2,npTrain3.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasIII += np.sum(bias3)
		
		variance3 = (np.dot(g2List[i],npTrain3.transpose()) - np.dot(g2,npTrain3.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceIII += np.sum(variance3)
		
		dataFrameTemp["col" + str(2)] = dataFrameTemp[i]**2
		npTrain4 = np.array(dataFrameTemp)
		bias4 = ((np.dot(g3,npTrain4.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasIV += np.sum(bias4)

		variance4 = (np.dot(g3List[i],npTrain4.transpose()) - np.dot(g3,npTrain4.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceIV += np.sum(variance4)
		
		dataFrameTemp["col" + str(3)] = dataFrameTemp[i]**3
		npTrain5 = np.array(dataFrameTemp)
		bias5 = ((np.dot(g4,npTrain5.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasV += np.sum(bias5)

		variance5 = (np.dot(g4List[i],npTrain5.transpose()) - np.dot(g4,npTrain5.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceV += np.sum(variance5)

		dataFrameTemp["col" + str(4)] = dataFrameTemp[i]**4
		npTrain6 = np.array(dataFrameTemp)
		bias6 = ((np.dot(g5,npTrain6.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasVI += np.sum(bias6)

		variance6 = (np.dot(g5List[i],npTrain6.transpose()) - np.dot(g5,npTrain6.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceVI += np.sum(variance6)
	print "\n-----------------------------------------------------------------------------\n"	
	print "Ridge Regression for Sample Size " + str(N) + " and lambda value " + str(lambdaValue)
	#print "\n-----------------------------------------------------------------------------\n"
	# print "\nPlotting the Histograms"	
	# print "Close all the histogram plots to continue with the program execution"
	# for i in range(len(dataFrameError)):
	# 	dataFrameError.iloc[i].hist()
		
	# 	plot.show()
	#print "\n-----------------------------------------------------------------------------\n"
	#print "\nSum Square Error"
	#print list(dataFrameError.iloc[2] * 10)
	#print "\n-----------------------------------------------------------------------------\n"
	#print "\nBias\n"
	# print "Bias of g1 " + str(biasI/100)
	# print "Bias of g2 " + str(biasII/100)
	# print "Bias of g3 " + str(biasIII/100)
	print "\nBias^2 of g4 " + str(biasIV/100)
	# print "Bias of g5 " + str(biasV/100)
	# print "Bias of g6 " + str(biasVI/100)
	#print "\n-----------------------------------------------------------------------------\n"
	#print "\nVariance\n"
	# print "Variance of g1 " + str(varianceI/100)
	# print "Variance of g2 " + str(varianceII/100)
	# print "Variance of g3 " + str(varianceIII/100)
	print "\nVariance of g4 " + str(varianceIV/100)
	# print "Variance of g5 " + str(varianceV/100)
	# print "Variance of g6 " + str(varianceVI/100)

def ridgeRegression(dataFrameTrain,lambdaValue):
	meanFrameTrain = dataFrameTrain.mean()
	stdFrameTrain = dataFrameTrain.std()
	normalFrameTrain = (dataFrameTrain - meanFrameTrain)/stdFrameTrain
	normalFrameTrain.insert(0,'ONES',1)
	normalFrameTrain = normalFrameTrain.drop(['y'],1)
	targetValuesTrain = np.array(dataFrameTrain.y)
	numpyTrainMatrix = normalFrameTrain.as_matrix()
	identityMatrix = lambdaValue * np.identity(len(numpyTrainMatrix.transpose()))
	lmsTrainMatrix = np.dot(np.dot(np.linalg.pinv(np.dot(numpyTrainMatrix.transpose(),numpyTrainMatrix)+ identityMatrix),numpyTrainMatrix.transpose()), targetValuesTrain)
	predictionValuesTrain = []
	
	for i in range(len(numpyTrainMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTrainMatrix[i])
		predictionValuesTrain.append(d)
	trainError = ((predictionValuesTrain - targetValuesTrain) ** 2).mean()
	return lmsTrainMatrix, trainError


def biasVarianceLinear(N):
	dataFrameTrain= pd.DataFrame()
	dataFrameError = pd.DataFrame()	
	dataFrameTemp = pd.DataFrame()
	squareError =0
	g1List = []
	g2List = []
	g3List = []
	g4List = []
	g5List = []
	for i in range(100):
		errorList = []
		dataFrameTemp = pd.DataFrame()
		dataFrameTrain[i] = np.random.uniform(-1,1,N)
		ylist=[]
		
		for x in range(0,len(dataFrameTrain)):
			ylist.append(2*(dataFrameTrain.iloc[x][i]**2) + np.random.normal(0,math.sqrt(0.1)))
		dataFrameTrain['y'] = ylist
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp['y'] = dataFrameTrain['y']
		dataFrameTemp['y'] = 1 - dataFrameTemp['y']
		dataFrameTemp['y'] = dataFrameTemp['y'] ** 2
		errorList.append(dataFrameTemp['y'].mean())
		dataFrameTemp['y'] = dataFrameTrain['y']
		w, error = linearRegression(dataFrameTemp)
		
		g1List.append(w)
		errorList.append(error)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp['y'] = dataFrameTrain['y']
		w1, error = linearRegression(dataFrameTemp)
		
		g2List.append(w1)	
		errorList.append(error)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp['y'] = dataFrameTrain['y']
		dataFrameTemp["col" + str(2)] = dataFrameTemp[i]**2
		
		w2,error = linearRegression(dataFrameTemp)
		errorList.append(error)
		g3List.append(w2)
		
		
		dataFrameTemp["col" + str(3)] = dataFrameTemp[i]**3
		w3,error = linearRegression(dataFrameTemp)
		errorList.append(error)
		g4List.append(w3)
		
		dataFrameTemp["col" + str(4)] = dataFrameTemp[i]**4
		w4,error = linearRegression(dataFrameTemp)
		errorList.append(error)
		g5List.append(w4)


		dataFrameError[i] = (errorList)
	g1 = np.sum(g1List,axis=0)/100
	g2 = np.sum(g2List,axis=0)/100
	g3 = np.sum(g3List,axis=0)/100
	g4 = np.sum(g4List,axis=0)/100
	g5 = np.sum(g5List,axis=0)/100
	
	biasI =0
	biasII = 0
	biasIII = 0
	biasIV = 0
	biasV = 0
	biasVI = 0
	
	varianceI = 0
	varianceII = 0
	varianceIII = 0
	varianceIV = 0
	varianceV = 0
	varianceVI = 0

	for i in range(100):
		errorList = []
		ylist =[]
		xlist =[]
		dataFrameTemp = pd.DataFrame()
		dataFrameTrain[i] = np.random.uniform(-1,1,N)
			
		for x in range(0,len(dataFrameTrain)):
			ylist.append(2*(dataFrameTrain.iloc[x][i]**2) + np.random.normal(0,math.sqrt(0.1)))
			xlist.append(dataFrameTrain.iloc[x][i])
		dataFrameTrain['y'] = ylist
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp = dataFrameTrain.copy(deep = True)
		dataFrameTemp.insert(0,'ONES',1)
		dataFrameTemp['y'] = dataFrameTemp['ONES'] - dataFrameTemp['y']
		npTrain1 = np.array(dataFrameTemp['y'])
		bias1 = (((npTrain1))**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		variance1 = 1 - dataFrameTemp['ONES'].mean()
		varianceI += variance1
		biasI += np.sum(bias1)
		
		dataFrameTemp =pd.DataFrame()
		dataFrameTemp = dataFrameTrain.copy(deep=True)
		dataFrameTemp.insert(0,'ONES',1)
		dataFrameTemp['vals'] = dataFrameTemp['ONES'] * g1
		
		npTrain2 = np.array(dataFrameTemp['vals'])
		npVariance2 = np.array(dataFrameTemp['ONES'])
		
		bias2 = ((npTrain2 - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasII += np.sum(bias2)
		
		variance2 = ((g1List[i]* npVariance2) -  (npTrain2))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceII += np.sum(variance2)

		dataFrameTrain = dataFrameTrain.drop(['y'],1)
		dataFrameTemp = pd.DataFrame()
		dataFrameTemp[i] = dataFrameTrain[i]
		dataFrameTemp.insert(0,'ONES',1)
		
		npTrain3 = np.array(dataFrameTemp)
		bias3 = ((np.dot(g2,npTrain3.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasIII += np.sum(bias3)
		
		variance3 = (np.dot(g2List[i],npTrain3.transpose()) - np.dot(g2,npTrain3.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceIII += np.sum(variance3)
		
		dataFrameTemp["col" + str(2)] = dataFrameTemp[i]**2
		npTrain4 = np.array(dataFrameTemp)
		bias4 = ((np.dot(g3,npTrain4.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasIV += np.sum(bias4)

		variance4 = (np.dot(g3List[i],npTrain4.transpose()) - np.dot(g3,npTrain4.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceIV += np.sum(variance4)
		
		dataFrameTemp["col" + str(3)] = dataFrameTemp[i]**3
		npTrain5 = np.array(dataFrameTemp)
		bias5 = ((np.dot(g4,npTrain5.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasV += np.sum(bias5)

		variance5 = (np.dot(g4List[i],npTrain5.transpose()) - np.dot(g4,npTrain5.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceV += np.sum(variance5)

		dataFrameTemp["col" + str(4)] = dataFrameTemp[i]**4
		npTrain6 = np.array(dataFrameTemp)
		bias6 = ((np.dot(g5,npTrain6.transpose()) - ylist)**2) * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		biasVI += np.sum(bias6)

		variance6 = (np.dot(g5List[i],npTrain6.transpose()) - np.dot(g5,npTrain6.transpose()))**2 * 1/N * norm.pdf(ylist,2*(np.array(xlist)**2), math.sqrt(0.1))
		varianceVI += np.sum(variance6)
	print "\n-----------------------------------------------------------------------------\n"	
	print "Linear Regression for Sample Size " + str(N)
	#print "\n-----------------------------------------------------------------------------\n"
	print "\nPlotting the Histograms"	
	print "Close all the histogram plots to continue with the program execution"
	for i in range(len(dataFrameError)):
		dataFrameError.iloc[i].hist()
		
		plot.show()
	print "\n-----------------------------------------------------------------------------\n"
	#print "\nSum Square Error"
#	print list(dataFrameError.iloc[2] * 10)
	#print "\n-----------------------------------------------------------------------------\n"
	print "Bias^2\n"
	print "Bias^2 of g1 " + str(biasI/100)
	print "Bias^2 of g2 " + str(biasII/100)
	print "Bias^2 of g3 " + str(biasIII/100)
	print "Bias^2 of g4 " + str(biasIV/100)
	print "Bias^2 of g5 " + str(biasV/100)
	print "Bias^2 of g6 " + str(biasVI/100)
	#print "\n-----------------------------------------------------------------------------\n"
	print "\nVariance\n"
	print "Variance of g1 " + str(varianceI/100)
	print "Variance of g2 " + str(varianceII/100)
	print "Variance of g3 " + str(varianceIII/100)
	print "Variance of g4 " + str(varianceIV/100)
	print "Variance of g5 " + str(varianceV/100)
	print "Variance of g6 " + str(varianceVI/100)
		
def linearRegression(dataFrameTrain):
	#print dataFrameTrain
	meanFrameTrain = dataFrameTrain.mean()
	stdFrameTrain = dataFrameTrain.std()
	normalFrameTrain = (dataFrameTrain - meanFrameTrain)/stdFrameTrain
	normalFrameTrain.insert(0,'ONES',1)
	normalFrameTrain = normalFrameTrain.drop(['y'],1)
	targetValuesTrain = np.array(dataFrameTrain.y)
	numpyTrainMatrix = normalFrameTrain.as_matrix()
	lmsTrainMatrix = np.dot(np.dot(np.linalg.pinv(np.dot(numpyTrainMatrix.transpose(),numpyTrainMatrix)),numpyTrainMatrix.transpose()), targetValuesTrain)
	predictionValuesTrain = []
	for i in range(len(numpyTrainMatrix)):
		d = np.dot(lmsTrainMatrix,numpyTrainMatrix[i])
		predictionValuesTrain.append(d)
	colDifferenceTrain = predictionValuesTrain - targetValuesTrain
	
	trainMeanSquareError = (colDifferenceTrain ** 2).mean()
	return lmsTrainMatrix, trainMeanSquareError

def main():
	biasVarianceLinear(10)
	biasVarianceLinear(100)
	lambdaValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
	for i in range(len(lambdaValues)):
		biasVarianceRidge(100,lambdaValues[i])

if __name__ == '__main__':
	import sys
	main()
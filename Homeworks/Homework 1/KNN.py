import pandas as pd
import numpy as np
import operator
import math
from collections import Counter
def getData():
	global dataFrame,arrTrain,column,mu,sigma
	dataFrame = pd.read_csv('train.txt', header =None)
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	dataFrame.columns =column
	arrTrain = list(dataFrame.values.tolist())
	mu = {}
	sigma = {}
	for col in column:
		mu[col] = dataFrame[col].mean()
		sigma[col] = dataFrame[col].std()
	
def getEucledianDistance(arrTests,arrTrains):
	distance = 0
	for y in column[1:-1]:
			distance += pow(((arrTests[y]) - (arrTrains[y])),2)
	return math.sqrt(distance)
def getManhattanDistance(arrTests,arrTrains):
	distance = 0
	for y in range(1,len(column)-1):
			distance += abs((arrTests[y])- (arrTrains[y]))
	return (distance)
def getNeighbors(fileType): 
	global dataFrameTest,arrTest,eucledianDistance,manhattanDistance
	if fileType == 0:
		dataFrameTest = pd.read_csv('test.txt', header =None)
	else:
		dataFrameTest = pd.read_csv('train.txt', header =None)
	#column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	dataFrameTest.columns =column
	arrTest = list(dataFrameTest.values.tolist())
	eucledianDistance = [[0]*len(dataFrame) for i in range(len(dataFrameTest))]
	manhattanDistance = [[0]*len(dataFrame) for i in range(len(dataFrameTest))]
	for i in range(len(dataFrameTest)):
		for j in range(len(dataFrame)):
			if dataFrame.equals(dataFrameTest):
				if i != j:
					eucledianDistance[i][j] = getEucledianDistance(dataFrameTest.loc[i],dataFrame.loc[j])
					manhattanDistance[i][j] = getManhattanDistance(dataFrameTest.loc[i],dataFrame.loc[j])
					
				else:
					eucledianDistance[i][j] = {"distance" : float("inf"), "class" : int(dataFrame.loc[j]['Class'])}
					manhattanDistance[i][j] = {"distance" : float("inf"), "class" : int(dataFrame.loc[j]['Class'])}
			else:
				eucledianDistance[i][j] = getEucledianDistance(dataFrameTest.loc[i],dataFrame.loc[j])
				manhattanDistance[i][j] = getManhattanDistance(dataFrameTest.loc[i],dataFrame.loc[j])
		
def getKNeighbors(distance):
	indices=[]
	classPrediction = []
	check = []
	k=1
	while k<9:
		indices=[]
		classPrediction = []
		for x in range(len(dataFrameTest)):
			npArray = np.array(distance[x])
			minK = npArray.argsort()[:k]
			check = []
			for i in range(minK.size):
				check.append(dataFrame.iloc[minK[i]].Class)
			count=Counter(check)
			repeatedEqual=count.most_common()
			if repeatedEqual[0][1] > math.ceil(k/2):
				minValue = repeatedEqual[0][0]
			else:
				tempMin = 1
				distList = []
				for g in range(len(repeatedEqual)-1):
				 if repeatedEqual[g][1] > repeatedEqual[g + 1][1]:
				 	tempMin = repeatedEqual[1][0]
				 	break;
				 elif repeatedEqual[g][1] == repeatedEqual[g+1][1]:
				 	for length in range(minK.size):
				 		if dataFrame.iloc[minK[length]].Class == repeatedEqual[g][0] or dataFrame.iloc[minK[length]].Class ==repeatedEqual[g+1][0]:
				 			distList.append(minK[length])
				 	tempMin = dataFrame.iloc[min(distList)].Class
				 minValue =tempMin
			classPrediction.append(minValue)
		print "\nK = " + str(k)
		acc =  [i for i, j in zip(classPrediction, list(dataFrameTest["Class"])) if i == j]
		print str(round(float(len(acc)) / float(len(list(dataFrameTest["Class"]))) * 100,2)) + "%"
		k = k+2
def main():
	getData() 
	print "\nTesting Accuracy"
	getNeighbors(0)
	print "\nKNN Accuracy for Eucledian Distance"
	getKNeighbors(eucledianDistance)
	print "\nKNN Accuracy for Manhattan Distance"
	getKNeighbors(manhattanDistance)
	print "\nTraining Accuracy"
	getNeighbors(1)
	print "KNN Accuracy for Eucledian Distance"
	getKNeighbors(eucledianDistance)
	print "\nKNN Accuracy for Manhattan Distance"
	getKNeighbors(manhattanDistance)
if __name__ == '__main__':
	import sys
	main()
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
from scipy.stats import multivariate_normal
import itertools as it
import random
from timeit import default_timer
from mpl_toolkits.mplot3d import Axes3D

def getData(fileName):
	global column, targetValuesTrain,targetValuesTest
	dataFrameTrain = pd.read_csv(fileName, header =None)
	column = ['x','y']
	dataFrameTrain.columns = column
	#print dataFrameTrain
	return dataFrameTrain 

def plotCluster(dataFrameTrain,posteriorFrame):
	for index,rows in posteriorFrame.iterrows():
		x = np.argmax(np.array(rows))
		if x == 0:
			plot.plot(dataFrameTrain.iloc[index]['x'],dataFrameTrain.iloc[index]['y'],'ro')
		elif x == 1:
			plot.plot(dataFrameTrain.iloc[index]['x'],dataFrameTrain.iloc[index]['y'],'go')
		elif x == 2:
			plot.plot(dataFrameTrain.iloc[index]['x'],dataFrameTrain.iloc[index]['y'],'bo')

	plot.show()


def gaussianModel(dataFrameTrain,k):
	bestLog = float('-inf')
	bestMean = []
	bestCovarianceI = []
	bestCovarianceII = []
	bestCovarianceIII = []
	bestPosterior = pd.DataFrame()
	count =0
	for h in range(5):
		meanValues = []
		prior = [1.0/3,1.0/3,1.0/3]
		
		for i in range(k):
			meanValues.append([])
			row = random.randint(0,len(dataFrameTrain))
			x = dataFrameTrain.iloc[row]['x']
			y = dataFrameTrain.iloc[row]['y']
			meanValues[i].append(x)
			meanValues[i].append(y)

		covarianceI = []
		covarianceII = []
		covarianceIII = []
		for index,rows in dataFrameTrain.iterrows():
			#covariancea.append([])
			value1 = (rows - meanValues[0]).as_matrix()
			covarianceI.append(list(value1))
					
			value2 = (rows - meanValues[1]).as_matrix()
			covarianceII.append(list(value2))
			
			value3 = (rows - meanValues[2]).as_matrix()
			covarianceIII.append(list(value3))
			
		covarianceI = np.dot(np.array(covarianceI).transpose(),np.array(covarianceI))/len(dataFrameTrain)
		covarianceII = np.dot(np.array(covarianceII).transpose(),np.array(covarianceII))/len(dataFrameTrain)
		covarianceIII = np.dot(np.array(covarianceIII).transpose(),np.array(covarianceIII))/len(dataFrameTrain)

		#repeats 5 times
		
		
		iterValues = [40,40,40,40,40]
		colors = ['r+-','g+-','b+-','k+-','m+-']
		logValues =[]
		zValues = []
		print "\n-----------------------------------------------------------------------------\n"
		print "Number of iterations " + str(iterValues[h])
		
		for z in range(iterValues[h]):
			conditionalI = []
			conditionalII = []
			conditionalIII = []
			for index,rows in dataFrameTrain.iterrows():
				conditionalI.append(multivariate_normal.pdf(rows,meanValues[0],covarianceI))
				conditionalII.append(multivariate_normal.pdf(rows,meanValues[1],covarianceII))
				conditionalIII.append(multivariate_normal.pdf(rows,meanValues[2],covarianceIII))
			conditionalFrame = pd.DataFrame({'c1':conditionalI,'c2':conditionalII,'c3':conditionalIII})
			#print conditionalFrame	
			posteriorI = []
			posteriorII = []
			posteriorIII = []
			logSum = 0
			for index,rows in conditionalFrame.iterrows():
				posteriorI.append(rows['c1']*prior[0] / (rows['c1']*prior[0] + rows['c2']*prior[1] + rows['c3']*prior[2]))
				posteriorII.append(rows['c2']*prior[1] / (rows['c1']*prior[0] + rows['c2']*prior[1] + rows['c3']*prior[2]))
				posteriorIII.append(rows['c3']*prior[2] / (rows['c1']*prior[0] + rows['c2']*prior[1] + rows['c3']*prior[2]))
				logSum += np.log((rows['c1']*prior[0] + rows['c2']*prior[1] + rows['c3']*prior[2]))

			posteriorFrame = pd.DataFrame({'p1':posteriorI,'p2':posteriorII,'p3':posteriorIII})
			
			logValues.append(logSum)
			zValues.append(z+1)
				
			meanI = 0
			meanII = 0
			meanIII = 0
			for index,rows in posteriorFrame.iterrows():
				meanI += rows['p1'] * dataFrameTrain.iloc[index]
				meanII += rows['p2'] * dataFrameTrain.iloc[index]
				meanIII += rows['p3'] * dataFrameTrain.iloc[index]
			meanValues[0] = list(meanI/posteriorFrame.p1.sum())
			meanValues[1] = list(meanII/posteriorFrame.p2.sum())
			meanValues[2] = list(meanIII/posteriorFrame.p3.sum())
			#print meanValues
			prior[0] = posteriorFrame.p1.sum() / len(dataFrameTrain)
			prior[1] = posteriorFrame.p2.sum() / len(dataFrameTrain)
			prior[2] = posteriorFrame.p3.sum() / len(dataFrameTrain)
			# print prior
			numerator1 = 0
			numerator2 = 0
			numerator3 = 0
			for index,rows in dataFrameTrain.iterrows():
				value1 = np.matrix(rows - meanValues[0])
				numerator1 += posteriorFrame.iloc[index].p1 * np.dot(value1.transpose(),value1)

				value2 = np.matrix(rows - meanValues[1])
				numerator2 += posteriorFrame.iloc[index].p2 * np.dot(value2.transpose(),value2)

				value3 = np.matrix(rows - meanValues[2])
				numerator3 += posteriorFrame.iloc[index].p3 * np.dot(value3.transpose(),value3)
					
			covarianceI = numerator1 / posteriorFrame.p1.sum()
			covarianceII = numerator2 / posteriorFrame.p2.sum()
			covarianceIII = numerator3 / posteriorFrame.p3.sum()
			#print logSum
	#	print bestLog
		if logSum > bestLog:
			count = iterValues[h]
			bestLog = logSum
			bestMean = meanValues
			bestCovarianceI = covarianceI
			bestCovarianceII = covarianceII
			bestCovarianceIII = covarianceIII
			bestPosterior = posteriorFrame
		plot.plot(zValues,logValues,colors[h])	
	print "Plot of the log likelihood of the data"
	print "Close the plot to continue with the program execution"
	plot.show()
	print "\n-----------------------------------------------------------------------------\n"
	print "Mean Values for k = 1"
	print bestMean[0]
	print "Mean Values for k = 2"
	print bestMean[1]
	print "Mean Values for k = 3"
	print bestMean[2]
	print "\n-----------------------------------------------------------------------------\n"
	print "Covariance Values for k = 1"
	print bestCovarianceI
	print "Covariance Values for k = 2"
	print bestCovarianceII
	print "Covariance Values for k = 3"
	print bestCovarianceIII
	print "\n-----------------------------------------------------------------------------\n"
	return bestPosterior

def main():
	dataFrameTrain = getData('hw5_blob.csv')
	posteriorFrame = gaussianModel(dataFrameTrain,3)
	plotCluster(dataFrameTrain,posteriorFrame)

if __name__ == '__main__':
	import sys
	main()
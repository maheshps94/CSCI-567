import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
import scipy as sp
import itertools as it
import sys
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
def getKernelDistance(x2y2,clusterCenters):
	distance = 0
	realDistance = []
	#print clusterCenters
	for i in range(len(clusterCenters)):
		distance += pow((x2y2 - clusterCenters[i][0]),2)
		distance = math.sqrt(distance)
		#distance = math.exp(-distance/0.0004)
		realDistance.append(distance)
		distance = 0

	return realDistance.index(min(realDistance))

def getEucledianDistance(x,y,clusterCenters):
	distance = 0
	realDistance = []
	#print clusterCenters
	for i in range(len(clusterCenters)):
		distance += pow((x - clusterCenters[i][0]),2)
		distance += pow((y - clusterCenters[i][1]),2)
		distance = math.sqrt(distance)
		realDistance.append(distance)
		distance = 0

	return realDistance.index(min(realDistance))

def plotClusters(clusters,k):
	df = pd.DataFrame(clusters["I"])
	df.columns = ['x','y']
	plot.plot(df.x, df.y, 'ro')
	df = pd.DataFrame(clusters["II"])
	df.columns = ['x','y']
	plot.plot(df.x, df.y, 'go')
	if k == 3 or k == 5:
		df = pd.DataFrame(clusters["III"])
		df.columns = ['x','y']
		plot.plot(df.x, df.y, 'bo')
	if k == 5:
		df = pd.DataFrame(clusters["IV"])
		df.columns = ['x','y']
		plot.plot(df.x, df.y, 'ko')
		df = pd.DataFrame(clusters["V"])
		df.columns = ['x','y']
		plot.plot(df.x, df.y, 'mo')
	plot.show()
def kernelPlotClusters(dataFrameTrain,k,clusterSet):
	#print clusterSet
	for index,data in dataFrameTrain.iterrows():
		if clusterSet[index] == 0:
			plot.plot(data['x'],data['y'], 'ro')
		elif clusterSet[index] == 1:
			plot.plot(data['x'],data['y'], 'go')
		elif clusterSet[index] == 2:
			plot.plot(data['x'],data['y'], 'bo')
		elif clusterSet[index] == 3:
			plot.plot(data['x'],data['y'], 'ko')
		elif clusterSet[index] == 4:
			plot.plot(data['x'],data['y'], 'mo')
	
	plot.show()

def kernelKMeans(dataFrameTrain, k):
	clusterCenters = []
	x = 0
	y = 0
	oldClusterSet = [-1] * len(dataFrameTrain)
	newClusterSet = []
	dataFrameTrain['x2'] = dataFrameTrain.x ** 2
	dataFrameTrain['y2'] = dataFrameTrain.y ** 2
	dataFrameTrain['x2y2'] = dataFrameTrain['x2'] + dataFrameTrain['y2']
	#print dataFrameTrain
	for i in range(k):
		clusterCenters.append([])
		row = random.randint(0,len(dataFrameTrain))
		x2y2 = dataFrameTrain.iloc[row]['x2y2']
		clusterCenters[i].append(x2y2)
		
	#print maxY
	#print clusterCenters
	while True:
		clusters = {'I':[],'II':[],'III':[],'IV':[],'V':[]}
		for index,data in dataFrameTrain.iterrows():
			#print data['x2']
			clusterNumber = getKernelDistance(data['x2y2'],clusterCenters)
			if clusterNumber == 0:
				clusters["I"].append([data['x2y2']])
			elif clusterNumber == 1:
				clusters["II"].append([data['x2y2']])
			elif clusterNumber == 2:
				clusters["III"].append([data['x2y2']])
			elif clusterNumber == 3:
				clusters["IV"].append([data['x2y2']])
			else:
				clusters["V"].append([data['x2y2']])
			newClusterSet.append(clusterNumber)
		
		newclusterCenters = []
		#newClusters = {'I':[],'II':[],'III':[],'IV':[],'V':[]}
		for i in range(k):
			newclusterCenters.append([])
			if i == 0:
				if len(clusters["I"]) == 0:
					clusters["I"].append(clusterCenters[i])
				x = np.mean(clusters["I"],axis = 0)
				newclusterCenters[i].append(x[0])
			if i == 1:
				if len(clusters["II"]) == 0:
					clusters["II"].append(clusterCenters[i])
				x = np.mean(clusters["II"],axis = 0)
				newclusterCenters[i].append(x[0])
			if i == 2:
				if len(clusters["III"]) == 0:
					clusters["III"].append(clusterCenters[i])
				x = np.mean(clusters["III"],axis = 0)
				newclusterCenters[i].append(x[0])
			if i == 3:
				if len(clusters["IV"]) == 0:
					clusters["IV"].append(clusterCenters[i])
				x = np.mean(clusters["IV"],axis = 0)
				newclusterCenters[i].append(x[0])
			if i == 4:
				if len(clusters["V"]) == 0:
					clusters["V"].append(clusterCenters[i])
				x = np.mean(clusters["V"],axis = 0)
				newclusterCenters[i].append(x[0])
		#print newclusterCenters
		if np.array_equal(oldClusterSet,newClusterSet):
			break
		else:
			oldClusterSet = newClusterSet
			newClusterSet = []
	#print newClusterSet
	kernelPlotClusters(dataFrameTrain,k,newClusterSet)
	#plotter(dataFrameTrain,k,newClusterSet)

def nearestPoints(dataFrameTrain, k):
	clusterCenters = []
	x = 0
	y = 0
	oldClusterSet = [-1] * len(dataFrameTrain)
	newClusterSet = []
	
	for i in range(k):
		clusterCenters.append([])
		row = random.randint(0,len(dataFrameTrain))
		x = dataFrameTrain.iloc[row]['x']
		y = dataFrameTrain.iloc[row]['y']
		clusterCenters[i].append(x)
		clusterCenters[i].append(y)
	#print maxY
	#print clusterCenters
	while True:
		clusters = {'I':[],'II':[],'III':[],'IV':[],'V':[]}
		for index,data in dataFrameTrain.iterrows():
			#print data['x']
			clusterNumber = getEucledianDistance(data['x'],data['y'],clusterCenters)
			if clusterNumber == 0:
				clusters["I"].append([data['x'],data['y']])
			elif clusterNumber == 1:
				clusters["II"].append([data['x'],data['y']])
			elif clusterNumber == 2:
				clusters["III"].append([data['x'],data['y']])
			elif clusterNumber == 3:
				clusters["IV"].append([data['x'],data['y']])
			else:
				clusters["V"].append([data['x'],data['y']])
			newClusterSet.append(clusterNumber)
		
		newclusterCenters = []
		#newClusters = {'I':[],'II':[],'III':[],'IV':[],'V':[]}
		for i in range(k):
			newclusterCenters.append([])
			if i == 0:
				if len(clusters["I"]) == 0:
					clusters["I"].append(clusterCenters[i])
				x = np.mean(clusters["I"],axis = 0)
				newclusterCenters[i].append(x[0])
				newclusterCenters[i].append(x[1])
			if i == 1:
				if len(clusters["II"]) == 0:
					clusters["II"].append(clusterCenters[i])
				x = np.mean(clusters["II"],axis = 0)
				newclusterCenters[i].append(x[0])
				newclusterCenters[i].append(x[1])
			if i == 2:
				if len(clusters["III"]) == 0:
					clusters["III"].append(clusterCenters[i])
				x = np.mean(clusters["III"],axis = 0)
				newclusterCenters[i].append(x[0])
				newclusterCenters[i].append(x[1])
			if i == 3:
				if len(clusters["IV"]) == 0:
					clusters["IV"].append(clusterCenters[i])
				x = np.mean(clusters["IV"],axis = 0)
				newclusterCenters[i].append(x[0])
				newclusterCenters[i].append(x[1])
			if i == 4:
				if len(clusters["V"]) == 0:
					clusters["V"].append(clusterCenters[i])
				x = np.mean(clusters["V"],axis = 0)
				newclusterCenters[i].append(x[0])
				newclusterCenters[i].append(x[1])
		#print newclusterCenters
		if np.array_equal(oldClusterSet,newClusterSet):
			break
		else:
			oldClusterSet = newClusterSet
			newClusterSet = []
	plotClusters(clusters,k)


def main():
	dataFrameTrain = getData('hw5_blob.csv')

	nearestPoints(dataFrameTrain,2)
	nearestPoints(dataFrameTrain,3)
	nearestPoints(dataFrameTrain,5)

	dataFrameTrain = getData('hw5_circle.csv')

	nearestPoints(dataFrameTrain,2)
	nearestPoints(dataFrameTrain,3)
	nearestPoints(dataFrameTrain,5)

	kernelKMeans(dataFrameTrain,2)
if __name__ == '__main__':
	import sys
	main()
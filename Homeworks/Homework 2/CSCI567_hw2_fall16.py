import LinearRegression as linear
import RidgeRegression as ridge

dataFrameTrain,dataFrameTest,pearsonSorted = linear.main()
ridge.main()
linear.secondMain(dataFrameTrain,dataFrameTest,pearsonSorted)

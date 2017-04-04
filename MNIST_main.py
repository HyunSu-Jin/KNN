import kNN

dataSet,labels = kNN.getTrainingDataSet_MNIST()
testDataSet,testLabels = kNN.getTestDataSet_MNIST()
print(kNN.getAccuracy(testDataSet,testLabels,dataSet,labels,3))
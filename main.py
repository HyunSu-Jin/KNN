import kNN

# Read .csv file
dataSet,labels = kNN.file2Matrix('datingTestSet.txt')

# determine test data ratio
testRatio = 0.1
total_m = dataSet.shape[0]
test = int(testRatio * total_m)

# Divide training dataSet & test dataSet

## test dataSet
testDataSet = dataSet[0:test,:]
testLables = labels[0:test]

## training dataSet
dataSet = dataSet[test::,:]
labels = labels[test::]

# find out the accuracy of classifier implemented by KNN (k = 3)
print("Accuracy : ",kNN.getAccuracy(testDataSet,testLables,dataSet,labels,3))
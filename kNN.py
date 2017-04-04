import numpy as np
import operator
import collections

def file2Matrix(filename):
    '''
    :param filename: filepath that we will read
    :return: datset, labels
    '''
    fr = open(filename)
    linelist = fr.readlines()
    featureNum = len(linelist[0].split('\t'))-1
    #print(featureNum)
    m = len(linelist)
    labels = []
    dataset = np.zeros((m,featureNum))
    index = 0
    for line in linelist:
        line = line.strip()
        tokens = line.split('\t')
        features = tokens[0:-1]
        dataset[index][:] = features
        labels.append(tokens[-1])
        index+=1
    return dataset, labels

def normalization(dataset):
    '''
    :param dataset: un-normalized dataset
    :return: normalized-dataset, minVals,ranges that used to normalize test-tuple
    '''
    minVals = dataset.min(axis=0) # get min values by column (vector)
    maxVals = dataset.max(axis=0) # get max values by column (vector)
    ranges = maxVals-minVals # get ranges (vector) It means max-min
    m = dataset.shape[0]
    normMatrix = dataset - np.tile(minVals,(m,1))
    normMatrix = normMatrix / np.tile(ranges,(m,1))
    return normMatrix, minVals,ranges

def predict(sample,normDataset,labels,minVals,ranges,k):
    '''
    input : non-normalized test-tuple, norm-matrix
    :return: the predicted label of test-tuple
    test-tuple로 정규화되어있지 않은 데이터를 받은뒤에, normalized된 dataSet으로 부터
    minVals,ranges를 참고하여 test-tuple를 normalize한다.
    '''
    # normalize the sample
    sample = sample - minVals
    sample = sample / ranges

    m = normDataset.shape[0] # the number of instances
    # calculate distance between testTuple and objects -- Euclid distance
    difMatrix = np.tile(sample,(m,1)) - normDataset # diff
    difMatrix = difMatrix ** 2 # square
    distances = difMatrix.sum(axis=1) # returns distances vector
    distances = distances ** 0.5
    ascendingIndice = distances.argsort() # distance가 작은것부터 오름차순으로 정렬
    classCount = collections.defaultdict(int) # default : 0
    for i in range(k):
        matchedLabel = labels[ascendingIndice[i]]
        classCount[matchedLabel] +=1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # operator.itemgetter(1) means order the dict by value
    # reverse=True means order the dict by descending order. default is ascending
    return sortedClassCount[0][0]

def classify(sample,dataset,lables,k):
    '''
    :param sample: test-tuple
    :param dataset: training dataSet
    :param lables: class label
    :param k: KNN count
    :return: predicted label
    이 함수는 정규화(normalized)되어 있지 않은 sample과 dataSet을 받아
    dataSet을 normalize하여 predict에 전달.
    '''
    normDataSet,minVals,ranges = normalization(dataset)
    prediction = predict(sample,normDataSet,lables,minVals,ranges,k)
    return prediction

def getAccuracy(testDataSet,testLables,dataSet,labels,k):
    test_m = testDataSet.shape[0]
    error_Cnt = 0
    index = 0
    for tuple in testDataSet:
        real_label = testLables[index]
        prediction = classify(tuple,dataSet,labels,k) # predict the label based on our dataset,labels
        if prediction != real_label:
            error_Cnt +=1
        index+=1
    print("Error count",error_Cnt)
    error_rate = error_Cnt / test_m
    accuacy = 1.0 - error_rate
    return accuacy
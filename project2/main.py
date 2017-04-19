from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: input,work, output
# TODO: ^ make a more descriptive todo
ITERATIONS = 100
LEARNINGRATE = 0.0000001


def main():
    (testData_x, testData_y, trainData_x, trainData_y) = importing_data()
    # part_one(testData_x, testData_y, trainData_x, trainData_y)
    # part_two(testData_x, testData_y, trainData_x, trainData_y)
    # part_three(testData_x, testData_y, trainData_x, trainData_y)


def part_one(testData_x, testData_y, trainData_x, trainData_y):
    print "part one"

    learningRate = 0.0000001
    #testing regression
    w = batchLogisticRegression(trainData_x, trainData_y,
                                learningRate)
    #print number of correctly predicted values
    correct = 0
    for i in range(testData_x.shape[0]):
        correct = correct + testLogisticRegression(testData_x[i,:], testData_y[i], w)

    percentageCorrect = correct/testData_x.shape[0]
    print percentageCorrect

def part_two(testData_x, testData_y, trainData_x, trainData_y):
    print "part two"

    learningRate = 0.0000001
    #testing regression
    accuracy = batchLogisticRegressionAccuracy(trainData_x, trainData_y,
                                learningRate, testData_x, testData_y)


    # Graph the percentages
    # graph_sse_x(accuracy[0], accuracy[1], np.arange(ITERATIONS), "percentage",
    #             "iterations")

def part_three(testData_x, testData_y, trainData_x, trainData_y):
    print "part three and four"
    lamb_range = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1,2,3,5,9, 10, 100, 1000]
    learningRate = 0.0000001
    test_accuracies = []
    train_accuracies = []
    for lamb in lamb_range:
        w = batchLogisticRegressionWithRegularization(trainData_x, trainData_y, learningRate, lamb)
        #print number of correctly predicted values
        # test data
        correct = 0
        for i in range(testData_x.shape[0]):
            correct = correct + testLogisticRegression(testData_x[i,:], testData_y[i], w)

        percentageCorrect = correct/testData_x.shape[0]
        test_accuracies.append(percentageCorrect)

        # # train data
        correct = 0
        for i in range(trainData_x.shape[0]):
            correct = correct + testLogisticRegression(trainData_x[i,:], trainData_y[i], w)

        percentageCorrect = correct/trainData_x.shape[0]
        train_accuracies.append(percentageCorrect)

    print lamb_range
    print test_accuracies
    graph_sse_x(train_accuracies, test_accuracies, lamb_range, "Test Accuracy", "Lambda")



def batchLogisticRegressionWithRegularization(x,y,lr,lamb):
    numFeatures = x.shape[1]
    w = np.zeros(numFeatures)
    j = 0
    while(1):
        d = np.zeros(numFeatures)
        for i, yi in enumerate(y):
            wxi = np.dot(w, x[i,:])
            nwxi = wxi * -1
            denom = 1 + np.exp(nwxi)
            regularization = np.multiply(lamb, w)
            yihat = 1/denom + regularization
            error = yi - yihat
            d = d + np.multiply(error, x[i,:])
        w = w + np.multiply(lr, d)
        j = j + 1
        # Break after j iterations
        if(j >= ITERATIONS):
            break;
    print w
    return w

# create w
# TODO: Fix Overflow
def batchLogisticRegression(x, y, lr):
    numFeatures = x.shape[1]
    w = np.zeros(numFeatures)
    j = 0
    while(1):
        d = np.zeros(numFeatures)
        for i, yi in enumerate(y):
            wxi = np.dot(w, x[i,:])
            nwxi = wxi * -1
            denom = 1 + np.exp(nwxi)
            yihat = 1/denom
            error = yi - yihat
            d = d + np.multiply(error, x[i,:])
        w = w + np.multiply(lr, d)
        j = j + 1
        # Break after j iterations
        if(j >= ITERATIONS):
            break;
    # print w
    return w

def batchLogisticRegressionAccuracy(x, y, lr, x2, y2):
    numFeatures = x.shape[1]
    resultsTrain = []
    resultsTest = []
    w = np.zeros(numFeatures)
    j = 0
    while(1):
        d = np.zeros(numFeatures)
        for i, yi in enumerate(y):
            wxi = np.dot(w, x[i,:])
            denom = 1 + np.exp(wxi * -1)
            yihat = 1/denom
            error = yi - yihat
            d = d + np.multiply(error, x[i,:])
        w = w + np.multiply(lr, d)
        j = j + 1

        (rTrain, rTest) = testAndRecord(x, y, x2, y2, w)
        resultsTrain.append(rTrain)
        resultsTest.append(rTest)

        # Stop after j iterations
        if(j >= ITERATIONS):
            break;
    return (resultsTrain, resultsTest)

def testAndRecord(x, y, x2, y2, w):
    #training accuracy
    correct = 0
    for i in range(x.shape[0]):
        correct = correct + testLogisticRegression(x[i,:], y[i], w)

    percentageCorrect = correct/x.shape[0]
    # print "training accuracy"
    # print percentageCorrect

    #testing accuracy
    correct2 = 0
    for i in range(x2.shape[0]):
        correct2 = correct2 + testLogisticRegression(x2[i,:], y2[i], w)

    percentageCorrect2 = correct2/x2.shape[0]
    # print "testing accuracy"
    # print percentageCorrect2

    return (percentageCorrect, percentageCorrect2)


# returns 1 if analysis was correct and 0 if analysis was incorrect
def testLogisticRegression(x, y, w):
    exp_wx = np.exp(np.matmul(np.transpose(w), x) * -1)

    denom_one = 1 + exp_wx
    prob_one = 1/denom_one

    prob_zero = exp_wx/denom_one
    if((prob_one/prob_zero) > 1):
        if(y == 1):
            return 1
    else:
        if(y == 0):
            return 1
    return 0

# def graph_sse_x(y, y1, x, ylabel, xlabel, ylegend, y1legend):
def graph_sse_x(y, y1, x, ylabel, xlabel):
    line1 = plt.plot(x, y)
    line2 = plt.plot(x, y1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.legend([ylegend,y1legend])

    plt.show()

# TODO: Test if this really parses data correctly
# TODO: (Possibly) creates a view of original, when view is modified so is
# original
def importing_data():
    print "importing"
    testLoc = '../project2/usps-4-9-test.csv'
    trainLoc = '../project2/usps-4-9-train.csv'

    dfTest = pd.read_csv(testLoc, header=None)
    dfTrain = pd.read_csv(trainLoc, header=None)

    dfTestArr = dfTest.values
    dfTrainArr = dfTrain.values

    (_, testSize_col) = dfTestArr.shape
    (_, trainSize_col) = dfTrainArr.shape

    dfTest_x = dfTestArr[:,:-1]
    dfTest_y = dfTestArr[:, testSize_col-1]

    dfTrain_x = dfTrainArr[:,:-1]
    dfTrain_y = dfTrainArr[:, trainSize_col-1]

    return (dfTest_x, dfTest_y, dfTrain_x, dfTrain_y)

def compute_w(x,y):
    ### compute optimal weight vector w
    xTx = np.matmul(np.transpose(x), x)
    xTx_inverse = np.linalg.inv(xTx)
    xTy = np.matmul(np.transpose(x), y)

    w = np.matmul(xTx_inverse, xTy)
    return w

if __name__ == '__main__':
    main()

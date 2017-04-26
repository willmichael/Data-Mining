from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    (testData_x, testData_y, trainData_x, trainData_y) = importing_data()
    print "test data: "
    print "x"
    print testData_x
    print "y"
    print testData_y
    "training data:"
    print "x"
    print trainData_x
    print "y"
    print trainData_y




def importing_data():
    print "Importing Data"
    testLoc = './knn_test.csv'
    trainLoc = './knn_train.csv'

    dfTest = pd.read_csv(testLoc, header=None)
    dfTrain = pd.read_csv(trainLoc, header=None)

    dfTestArr = dfTest.values
    dfTrainArr = dfTrain.values

    (_, testSize_col) = dfTestArr.shape
    (_, trainSize_col) = dfTrainArr.shape

    dfTest_x = dfTestArr[:,1:]
    dfTest_y = dfTestArr[:,0]
    dfTrain_x = dfTrainArr[:,1:]
    dfTrain_y = dfTrainArr[:,0]

    return (dfTest_x, dfTest_y, dfTrain_x, dfTrain_y)

if __name__ == '__main__':
    main()

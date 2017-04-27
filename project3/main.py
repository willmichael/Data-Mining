from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    (test_data, train_data) = importing_data()
    part_one(test_data, train_data)


def part_one(test_data, train_data):
    """
    Model selection for KNN
    """
    k = 5
    knn_algo(test_data, train_data, k)


def part_two():
    """
    Decision tree
    """
    pass

def part_three():
    """
    extra credit
    """
    pass

def knn_algo(test_data, train_data, k):
    # for each test item, find the distance between it and every item in the
    # training set
    for test_instance in test_data:
        distances = []
        for x in range(len(test_instance)):
            dist = euclideanDistance(test_instance, train_data[x])
            distances.append( (train_data[x], dist) )
        # print distances
        distances.sort(key=lambda tup: tup[1])

        # once all the distances have been found, find the K nearest neighbors
        neighbors = []
        for i in range(k):
            # store and print the neighbor's class
            neighbors.append(distances[i][0][0])
        print neighbors

def euclideanDistance(instance1, instance2):
    # find length of a instance
    length = len(instance1)
    # find distance between two instances
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    # print "distance: "
    # print distance
    return np.sqrt(distance)

def importing_data():
    print "Importing Data"
    testLoc = './knn_test.csv'
    trainLoc = './knn_train.csv'

    dfTest = pd.read_csv(testLoc, header=None)
    dfTrain = pd.read_csv(trainLoc, header=None)

    dfTrain_norm = (dfTrain - dfTrain.min()) / (dfTrain.max() - dfTrain.min())
    dfTest_norm = (dfTest - dfTest.min()) / (dfTest.max() - dfTest.min())

    dfTest = dfTrain_norm.values[:,:]
    dfTrain = dfTrain_norm.values[:,:]

    return (dfTest, dfTrain)

    

if __name__ == '__main__':
    main( )

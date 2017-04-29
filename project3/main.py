from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    (test_data, train_data) = importing_data()
    (range_k, train_acc, test_acc) = part_one(test_data, train_data)

    # plt.plot(range_k, train_acc)
    # plt.plot(range_k, test_acc)
    # plt.show()



def part_one(test_data, train_data):
    """
    Model selection for KNN
    """

    range_k = range(1, 5, 2)
    train_acc = []
    test_acc = []
    train_neighbors_dists = knn_algo(train_data, train_data)
    test_neighbors_dists = knn_algo(test_data, train_data)

    for k in range_k:
        # print "K: "
        # print k
        train_neighbors = find_k(train_neighbors_dists, k)
        train_err = (calculate_error(train_neighbors))

        test_neighbors = find_k(test_neighbors_dists, k)
        test_err = (calculate_error(test_neighbors))

        ### TODO: cross validation here
        train_acc.append(train_err)
        test_acc.append(test_err)

    print "Train acc: "
    print train_acc
    print "Test acc: "
    print test_acc
    print range_k

    return (range_k, train_acc, test_acc)

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


### needs tuple (1/0, [array of prediction values])
def calculate_error(test_neighbors):
    total = len(test_neighbors)
    correct = 0
    for tn in test_neighbors:
        if int(tn[0]) == vote_decision(tn[1]):
            correct += 1
        # else:
            # print "fail"
    # print correct
    # print total
    print "correct number: "
    print correct
    return 1-(correct/total)

def vote_decision(neighbors):
    num_ones = neighbors.count(1)
    num_zeros = neighbors.count(0)
    if num_ones > num_zeros:
        return 1
    elif num_zeros > num_ones:
        return 0
    else:
        print "error"
        return -1

def knn_algo_cross_validate(test_data, train_data, k):
    # for each test item, find the distance between it and every item in the
    # training set
    test_neighbors = []
    # j = 0
    for test_instance in test_data:
        distances = []
        for x in range(len(train_data)):
            dist = euclideanDistance(test_instance, train_data[x])
            # These distances are the all train distances for each particular test
            # instance
            # print dist
            distances.append( (train_data[x], dist) )
        # Sorted to find the lowest distance
        distances.sort(key=lambda tup: tup[1])
        # once all the distances have been found, find the K nearest neighbors
        neighbors = []

        # print test_instance[0]
        for i in range(k):
            # store and the lowest k neighbor's prediction value (1 or 0)
            neighbors.append(distances[i][0][0])

            # print distances[i]
        test_neighbors.append((test_instance[0], neighbors))
        # j += 1
        # if j == 100:
            # break
    return test_neighbors

def knn_algo(test_data, train_data):
    # for each test item, find the distance between it and every item in the
    # training set
    test_neighbors = []
    for test_instance in test_data:
        distances = []
        for x in range(len(train_data)):
            dist = euclideanDistance(test_instance, train_data[x])
            # These distances are the all train distances for each particular test
            # instance
            # print dist
            distances.append( (train_data[x], dist) )
        # Sorted to find the lowest distance
        distances.sort(key=lambda tup: tup[1])
        test_neighbors.append((test_instance, distances))
    return test_neighbors

def find_k(data, k):
    dataset = []
    for d in data:
        # once all the distances have been found, find the K nearest neighbors
        neighbors = []
        for i in range(k):
            # store and the lowest k neighbor's prediction value (1 or 0)
            neighbors.append(d[1][i][0][0])
        dataset.append((d[0][0], neighbors))
    return dataset

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

    dfTest = dfTest_norm.values[:,:]
    dfTrain = dfTrain_norm.values[:,:]

    return (dfTest, dfTrain)

if __name__ == '__main__':
    main( )

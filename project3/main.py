from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# TODO: input,work, output
# TODO: ^ make a more descriptive todo

class Tree(object):
    def __init__(self):
        self.data=None
        self.left=None
        self.right=None
        self.index=None

    def printTree(self, t):
        if t==None:
            return

        if t.left == None and t.right == None:
            print t.data
        self.printTree(t.left)
        self.printTree(t.right)

class dec_stump(object):
    def __init__(self, feature, root, left, right, entropy):
        self.feature = feature
        self.root = root
        self.leftInfo = left
        self.rightInfo = right
        self.left = 0 if left[0] > left[1] else 1
        self.right = 0 if right[0] > right[1] else 1
        self.entropy = entropy

    def predict(self, val):
        if val == 0:
            return self.left
        elif val == 1:
            return self.right
        else:
            print "error"
            return -1

    def print_stump(self):
        print "For feature: " + str(self.feature)
        print "Left Branch: " + str(self.left)
        print "Right Branch: " + str(self.right)
        print "Entropy: " + str(self.entropy)

    def print_stump_detailed(self):
        print "For feature: " + str(self.feature)
        print "Left Branch: " + str(self.leftInfo)
        print "Right Branch: " + str(self.rightInfo)
        print "Entropy: " + str(self.entropy)

def main():
    (test_data, train_data) = importing_data()
    (range_k, train_acc, test_acc, cv_acc) = part_one(test_data, train_data)
    part_two(test_data, train_data)

def part_one(test_data, train_data):
    """
    Model selection for KNN
    """

    range_k = range(1,71, 2)
    train_acc = []
    test_acc = []
    cv_acc = []
    train_neighbors_dists = knn_algo(train_data, train_data)
    test_neighbors_dists = knn_algo(test_data, train_data)
    cv_dists = knn_algo_cross_validate(train_data)

    for k in range_k:
        train_neighbors = find_k(train_neighbors_dists, k)
        train_err = (calculate_error(train_neighbors))

        test_neighbors = find_k(test_neighbors_dists, k)
        test_err = (calculate_error(test_neighbors))

        cv_neighbors = find_k(cv_dists, k)
        cv_err = (calculate_error(cv_neighbors))

        ### TODO: cross validation here
        train_acc.append(train_err)
        test_acc.append(test_err)
        cv_acc.append(cv_err)

    # part 1.1
    print "K range: "
    print range_k
    print "Train acc: "
    print train_acc
    print "Test acc: "
    print test_acc
    print "CV acc: "
    print cv_acc
    # part 1.2
    plt.plot(range_k, train_acc, label = "train")
    plt.plot(range_k, test_acc, label = "test")
    plt.plot(range_k, cv_acc, label = "CV")
    plt.ylabel("percent error")
    plt.xlabel("k")
    plt.legend()
    plt.show()
    # part 1.3
    print " We observe that as K get larger that the error rate rises. Our training \
        data is perfect when compared to itself and with a K of 1. The error rate \
        rises pretty rapidly there after. Our test data's error rate stays approximately \
        6 percent error for the first 70 K values. Our leave-one-out cross-validation \
        error rate has a somewhat linear climb similar to the training set. Looking at \
        the graph, a acceptable K value would be around the 35 K values."
    return (range_k, train_acc, test_acc, cv_acc)

def part_two(test_data, train_data):
    """
    Decision tree
    """
    # Create all stumps
    stump_list = create_stump(train_data)

    # Sort on entropy
    stump_list.sort(key=lambda x: x.entropy)

    # for sl in stump_list:
        # sl.print_stump_detailed()
        # print

    train_perc = []
    test_perc = []

    train_res = train_data[:, 0]
    test_res = test_data[:, 0]
    for i in range(1,10):
        train_col = train_data[:, i]
        test_col = test_data[:, i]

        print "Feature: " + str(stump_list[i-1].feature)

        countGood = 0
        for idx, trc in enumerate(train_col):
            half_train = calc_data_half(1, trc)
            if train_res[idx] == (stump_list[i-1].predict(half_train)):
                countGood += 1

        train_p = str(countGood/len(train_res)*100)
        train_perc.append(train_p)
        print "Train Correct Percentage: " + str(train_p)

        countGood = 0
        for idx, trc in enumerate(test_col):
            half_test = calc_data_half(1, trc)
            if test_res[idx] == (stump_list[i-1].predict(half_test)):
                countGood += 1

        test_p = str(countGood/len(test_res)*100)
        test_perc.append(test_p)
        print "Test Correct Percentage: " + str(test_p)

    decision_tree_with_depth(train_data, test_data)


def part_three():
    """
    extra credit
    """
    pass


def decision_tree_with_depth(train_data, test_data):
    # build the tree
    root = Tree()
    root.data = train_data
    head = root
    do_split(head, 6, 1)

    # With test data, test it againest the tree
    test_correct = 0
    for row in test_data:
        prediction = predict(head, row)
        # print "predicted: " + str(prediction) + " Expected: " + str(row[0])
        if prediction == row[0]:
            test_correct += 1

    # test train data againest tree
    train_correct = 0
    for row in train_data:
        prediction = predict(head, row)
        # print "predicted: " + str(prediction) + " Expected: " + str(row[0])
        if prediction == row[0]:
            train_correct += 1

    print "\nDecision tree with depth 6: "
    print "Train error: " + str((train_correct/len(test_data)*100))
    print "Test error: " + str((test_correct/len(test_data))*100)


def predict(root, row):
    if root == None:
        return None
    if len(root.data) == 1:
        return root.data[0]

    if row[root.index] < .5:
        return predict(root.left, row)
    else:
        return predict(root.right, row)


def test_split(attribute_index, head):
    head.left = Tree()
    head.left.data = []
    head.left.index = attribute_index
    head.right = Tree()
    head.right.data = []
    head.right.index = attribute_index
    for d in head.data:
        if d[attribute_index] < .5:
            head.left.data.append(d)
        else:
            head.right.data.append(d)
    return head

def find_split(root):
    if root is None:
        return root
    # find feature to split on
    features_range = range(1,30)
    entropies = []
    for i in features_range:
        test_split(i, root)

        root_data = data_into_nparray(root.data)
        rt = count_root(root_data)

        l_data = data_into_nparray(root.left.data)
        left = count_root(l_data)

        r_data = data_into_nparray(root.right.data)
        right = count_root(r_data)

        entropies.append(calc_entropy(rt, left, right))

    target_attribute = entropies.index(min(entropies)) +1
    return (min(entropies), target_attribute)

def make_leaf(head):
    if head is None or len(head.data) == 0:
        return None
    head.left = None
    head.right = None
    classifier = [row[0] for row in head.data]
    head.data = [max(set(classifier), key=classifier.count)]
    return head.data

def do_split(head, max_depth, depth):
    if head.data is None:
        return None

    # print "len of head data: " + str(len(head.data))
    # print [row[0] for row in head.data]

    if check_if_same_class(head.data):
        return make_leaf(head)

    if depth >= max_depth:
        make_leaf(head)
        return

    # do split
    target_index = find_split(head)
    head.index = target_index[1]
    test_split(target_index[1], head)

    # To print tree
    # print "\nDepth: " + str(depth)
    # print "Splitting by index #" + str(target_index[1])
    # print "Info gain: " + str(target_index[0])

    if head.left is None and head.right is None:
        return

    # go left
    if len(head.left.data) == 0:
        head.left = None
    elif len(head.left.data) <= 1:
        make_leaf(head.left)
    else:
        do_split(head.left, max_depth, depth+1)

    # go right
    if len(head.right.data) == 0:
        head.right = None
    elif len(head.right.data) <= 1:
        make_leaf(head.right)
    else:
        do_split(head.right, max_depth, depth+1)



def data_into_nparray(data):
    temp = []
    for nd in data:
        temp.append(nd[0])
    temp = np.array(temp)
    return temp

def check_if_same_class(data):
    unique = data[0][0]
    for d in data:
        if d[0] == unique:
            continue
        else:
            return False
    return unique

def create_stump(data):
    results = data[:, 0]
    feature_list = range(1, len(data[0]), 1)
    entropy_list = []
    stump_list = []
    for i in range(1, len(data[0])):
        data_col = data[:, i]
        root_sub = count_root(results)
        (zero_sub, one_sub) = count_zero_one(data_col, results)

        entropy = calc_entropy(root_sub, zero_sub, one_sub)
        stump = dec_stump(i, root_sub, zero_sub, one_sub, entropy)
        stump_list.append(stump)
    return stump_list

def maj_label(count):
    if count[0] > count[1]:
        return 0
    else:
        return 1

def count_zero_one(data_col, results):
    data_res = zip([calc_data_half(1, x) for x in data_col], results)
    data_res.sort(key=lambda x: x[0])

    total_left_zero = [x[1] for x in data_res if x[0] == 0].count(0)
    total_left_one = [x[1] for x in data_res if x[0] == 0].count(1)
    total_right_zero = [x[1] for x in data_res if x[0] == 1].count(0)
    total_right_one = [x[1] for x in data_res if x[0] == 1].count(1)
    return ((total_left_zero, total_left_one),(total_right_zero, total_right_one))

def count_root(results):
    results = results.tolist()
    total_zero = results.count(0)
    total_one = results.count(1)
    return(total_zero, total_one)

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
    return (1-(correct/total)) * 100

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

def knn_algo_cross_validate(train_data):
    # for each test item, find the distance between it and every item in the
    # training set
    test_neighbors = []
    for test_instance in train_data:
        distances = []
        for x in range(len(train_data)):
            if np.array_equal(test_instance, train_data[x]):
                continue
            else:
                dist = euclideanDistance(test_instance, train_data[x])
                # These distances are the all train distances for each particular test
                # instance
                # print dist
                distances.append( (train_data[x], dist) )
        # Sorted to find the lowest distance
        distances.sort(key=lambda tup: tup[1])
        test_neighbors.append((test_instance, distances))
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
    for x in range(1,length):
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

def calc_data_half(max_val, data):
    half = max_val/2
    if data > half:
        return 1
    else:
        return 0

def calc_threshold_theta(data_col, results):
    data_res = zip(results, data_col)
    data_res.sort(key=lambda x: x[1])

    infoGain = []
    prevClass = 0
    for idx, dr in enumerate(data_res):
        # Calc entrop when class label changes
        if prevClass != dr[idx][0]:
            root_sub = count_root(results)
            (zero_sub, one_sub) = count_zero_one(data_col, results)
            entrop = calc_entropy(root_sub, zero_sub, one_sub)
            infoGain.append(entrop, dr[idx][1])
    
    # Sort on entropy to find minimum entropy
    entrop.sort(key=lambda x: x[0])
    # return the number we should be splitting on
    return entrop[0][1]

### accepts three tuples, root, left leaf, and right leaf
def calc_entropy(root_sub, one_sub, two_sub):
    root_uncertainty = node_uncertainty(root_sub)

    root_l = root_sub[0]/(root_sub[0] + root_sub[1])
    root_r = root_sub[1]/(root_sub[0] + root_sub[1])

    entropy = root_uncertainty - (root_l * node_uncertainty(one_sub)) - (root_r * node_uncertainty(two_sub))
    ### rounding errors
    if entropy < 0:
        return 0
    return entropy

def node_uncertainty(node):
    node_total = node[0] + node[1]
    if node_total == 0:
        return 0
    node_l = node[0]/node_total
    node_r = node[1]/node_total
    if node_l == 0 or node_r == 0:
        return 0
    node_uncertainty = (-1 * node_l) * np.log2(node_l) + (-1 * node_r) * np.log2(node_r)
    return node_uncertainty





if __name__ == '__main__':
    main( )

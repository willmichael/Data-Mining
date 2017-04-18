import numpy as np
import pandas as pd

# TODO: input,work, output
# TODO: ^ make a more descriptive todo

def main():
    (testData_x, testData_y, trainData_x, trainData_y) = importing_data()
    part_one(testData_x, testData_y, trainData_x, trainData_y)



def part_one(testData_x, testData_y, trainData_x, trainData_y):
    print "part one"

    #testing regression
    w = batchLogisticRegression(testData_x[380:420,:], testData_y[380:420])
    #print number of correctly predicted values
    correct = 0
    for i in range(80):
        correct = correct + testLogisticRegression(testData_x[380+i,:], testData_y[380+i], w)
    print correct

# create w 
def batchLogisticRegression(x, y):
    learningRate = .05
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

        w = w + np.multiply(learningRate, d)
        j = j + 1
        if(j > 100):
            break;

    # print w
    return w

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
    return -1
    # print trainData_x
    print len(trainData_x)
    print len(trainData_y)
    # np.savetxt('foo.csv', trainData_x, fmt='%10.5f', delimiter=',')

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

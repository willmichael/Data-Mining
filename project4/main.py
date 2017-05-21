
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    print "hi"

def importing_data():
    print "Importing Data"
    data = 'data-1.txt'

    data = pd.read_csv(data, header=None)
    dfTrain = pd.read_csv(trainLoc, header=None)


    dfData = data.values[:,:]

    return dfData


if __name__ == '__main__':
    main()

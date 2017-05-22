
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    print "hi"
    data = importing_data()
    print len(data)

def part_2_1():
    # implement k means with k = 2
    pass

def part_2_2():
    # implement k means with k = range(2,11)
    pass

def part_3_1():
    # Implement HAC algorithm using single link to measure the distance
    # between clusters
    pass

def part_3_2():
    # Implement HAC algorithm using complete link to measure the distance
    # between clusters.
    pass


def importing_data():
    print "Importing Data"
    data = 'data-1.txt'
    data = pd.read_csv(data, header=None)
    dfData = data.values[:,:]
    return dfData


if __name__ == '__main__':
    main()

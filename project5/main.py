
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    (n, m, transitions_matrixes, rewards) =importing_data()
    print rewards
    print transitions_matrixes[0]

def importing_data():
    print "Importing Data"
    filename = 'test-data-for-MDP-1.txt'
    with open(filename, 'r') as f:
        first_line = f.readline().split()
        num_states = int(first_line[0]) # n
        num_actions = int(first_line[1]) # m
        transitions_matrixes = [] # list of m nxn matrixes

        # skip first blank line
        f.readline()
        # read m things
        for i in range(num_actions):
            matrix = []
            # read n lines
            for j in range(num_states):
                line = f.readline().split()
                matrix.append(line)
            transitions_matrixes.append(matrix)
            # skip blank line in between:
            f.readline()
        rewards = f.readline()
        ## print debugging
        # for i in transitions_matrixes:
        #     print str(i)
        #     print str(len(i))
        # print len(transitions_matrixes)
        return(num_states, num_actions, transitions_matrixes, rewards)

if __name__ == '__main__':
    main()

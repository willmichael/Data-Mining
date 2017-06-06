# Alex Nguyen, Michael Lee
# CS434 Machine Learning
# Spring 2017
# Homework 5

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    (num_states, num_actions, transitions_matrixes, rewards) = importing_data()
    # turn str to floats
    rewards = [float(i) for i in rewards]

    # part 2
    discount_factor = 0.1
    U = value_iteration_algo(num_states,num_actions, discount_factor, transitions_matrixes, rewards)
    P = calc_policy(num_states, num_actions, transitions_matrixes, U)
    print "Iterations: " + str(len(U))
    print "U: " + str(U)
    print "P: " + str(P)

    discount_factor = 0.9
    U = value_iteration_algo(num_states,num_actions, discount_factor, transitions_matrixes, rewards)
    P = calc_policy(num_states, num_actions, transitions_matrixes, U)
    print "Iterations: " + str(len(U))
    print "U: " + str(U)
    print "P: " + str(P)

# calculates optimal policy
def calc_policy(num_states, num_actions, transitions_matrixes, U):
    p = []
    # for each state
    for current_state in range(num_states):
        actions = []
        # for each action
        for a in range(num_actions):
            total = 0.0
            # for each  state calculate summation
            for target_state in range(num_states):
                total += float(transitions_matrixes[a][current_state][target_state]) * float(U[target_state])
            actions.append(total)
        # for all actions, find the argMax
        policy = actions.index(max(actions))
        # add to list of policies
        p.append(policy)
    return p

# wrapper for calculation
def value_iteration_algo(num_states,num_actions, discount_factor, transitions_matrixes, rewards):
    print "\nDoing algo ... "
    # init state, action
    curr_state = 0
    U = []
    step = 0
    print "discount_factor: " + str(discount_factor)

    # get reward for step 1 on
    U.append(rewards)
    while(1):
        # print "Step " + str(step)
        # print "U: " + str(U)
        # print "P: " + str(P)
        Ui = []
        policies = []
        # for each state, find Utility and policies
        for i in range(num_states):
            utility = calc_utility_for_one_state(num_actions,num_states, discount_factor, transitions_matrixes, rewards, i, U[step])
            Ui.append(utility)
        U.append(Ui)

        # check if we need to quit
        if step > 0:
            if check_delta(U, step, discount_factor):
                break
        step += 1

    # return last U
    # print "U: " + str(U)
    return U[-1]


# returns the sum of the future reward for going to target_state
def calc_utility_for_one_state(num_actions, num_states, discount_factor, transitions_matrix, rewards, current_state, Ui):
    actions = []
    # for all actions...
    for action in range(num_actions):
        total = 0.0
        for target_state in range(num_states):
            # print "working on state "+ str(current_state) +" going to state " + str(target_state) + " on action " + str(action)
            prob = transitions_matrix[action][current_state][target_state]
            expected_reward = Ui[target_state]
            total += float(prob) * float(expected_reward)
        actions.append(total)
        # print "actions: "
        # print actions
    Rs = rewards[current_state]
    utility = float(Rs) + float(discount_factor) * max(actions)
    # print "actions: " + str(actions)
    return utility

# checks the delta from the current step and the last step for
# returns true if delta is lower than threshold
def check_delta(U, step, discount_factor):
    # we set e to be really small here:
    e = 0.0000000001
    # find threshold
    top = e * ((1-discount_factor) ** 2)
    bottom = 2 * (discount_factor ** 2)
    threshold = top / bottom

    # sum diff of deltas from last iteration
    delta_total = 0.0
    for i in range(len(U[step])):
        delta = U[step][i] - U[step-1][i]
        delta_total += delta
    # print "delta total: " + str(delta_total)
    # print "threshold: " + str(threshold)

    # if delta total is less than threshold, return True
    if delta_total < threshold:
        return True
    else:
        return False


def importing_data():
    print "Importing Data..."
    filename = 'test-data-for-MDP-1.txt'
    # filename = "sample_data.txt"
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
        rewards = f.readline().split()
        ## print debugging
        # for i in transitions_matrixes:
        #     print str(i)
        #     print str(len(i))
        # print len(transitions_matrixes)
    return(num_states, num_actions, transitions_matrixes, rewards)

if __name__ == '__main__':
    main()

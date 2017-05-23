
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

# TODO: input,work, output
# TODO: ^ make a more descriptive todo


def main():
    data = importing_data()
    # sse = part_2_1(data)
    part_2_2(data)

def part_2_1(data):
    # implement k means with k = 2
    random.seed()

    
    ### algo
    # select k random samples of sampels
    # for each xi to cj such that distance from x is minimized
    # update cluster center
    # until convergence
    sse = k_means(data,2,15)
    plt.plot(sse)
    plt.show()
    return sse


def k_means(data, k, iterate):
    # pick seeds
    seednums = []
    for i in range(k):
        seednums.append(random.randint(0, len(data)-1))
    seeds = []
    for i in seednums:
        seeds.append(data[i])
    
    total_sse = []

    for it in range(iterate):
        # create k clusters
        clusters = create_clusters(seeds, data)
        centers = []

        # get new centers for each cluster
        for c in range(len(clusters)):
            centers.append(calc_center(clusters[c]))

        # calculate the sse
        sse_it = sse(clusters, centers)
        # print sse_it
        total_sse.append(sse_it)

        seeds = centers

    return total_sse


def create_clusters(seeds, data):
    # create clusters
    clusters = []
    for i in seeds:
        clusters.append([i])

    # assign clusters
    for point in data:
        distances = []
        # calculate distances
        for i,s in enumerate(seeds):
            if np.all(point == s):
                continue
            else:
                distances.append(euclideanDistance(point, s))

        # find idx of min distance
        idx = distances.index(min(distances))
        clusters[idx].append(point)
    
    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            print "cluster empty"
            print i
            print len(seeds)
            print seeds[i]
            print len(clusters)
    return clusters

def calc_center(cluster):
    n = len(cluster)
    if n == 0:
        print "found 0 cluster"
        print cluster
        print n
        return 0
    row_num = len(cluster[0])
    cluster = np.array(cluster)

    center = cluster.sum(axis=0)
    center = center/n
    return center

def sse(clusters, centers):
    sse = 0
    # for each cluster
    for i, cn in enumerate(centers):
        len_cn = len(cn)

        # find the difference from the center
        for clus in clusters[i]:
            # add to the sse
            for x in range(0,len_cn):
                sse += pow((clus[x] - cn[x]), 2)
    return sse

def euclideanDistance(instance1, instance2):
    # find length of a instance
    length = len(instance1)
    # find distance between two instances
    distance = 0
    for x in range(0,length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return np.sqrt(distance)

def part_2_2(data):
    # implement k means with k = range(2,11)
    sse_k = []
    for k in range(2,11):
        print "Doing K: " + str(k)
        sse = k_means(data, k, 10)
        sse_k.append(sse)
        line = plt.plot(sse, label="K = "+str(k))
        plt.legend()
    plt.show()

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
    data = 'data-2.txt'
    data = pd.read_csv(data, header=None)
    dfData = data.values[:,:]
    return dfData

if __name__ == '__main__':
    main()

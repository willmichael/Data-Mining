
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
    # part_2_2(data)
    part_3_2(data)

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

def part_2_2(data):
    # implement k means with k = range(2,11)
    sse_k = []
    for k in range(2,11):
        print "Doing K: " + str(k)
        sse = k_means(data, k, 10)
        sse_k.append(sse)
        plt.plot(sse)

    plt.show()

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
        clusters.append([])

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
        # print "len clusters " + str(i) + ": " + str(len(clusters[i]))
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

def part_3_1(data):
    # Implement HAC algorithm using single link to measure the distance
    # between clusters

    # Start with all objects in their own cluster.
    all_clusters = []
    for d in data:
        all_clusters.append([d])
    # Repeat until there is only one cluster:
    ## Among the current clusters, determine the clusters, ci and cj, that are closest
    ## Replace ci and cj with a single cluster
    length = len(all_clusters)
    while(length > 0):
        print " length: " + str(length)

        # for each cluster
        cluster_matrix = [[9999 for x in range(length)] for y in range(length)]
        for i in range(length):
            # TODO: implement skip list
            # compute best min distance from cluster to cluster
            cluster_distances = []
            # for each other cluster
            for j in range(length):
                # skip self
                if i == j:
                    cluster_matrix[i][j] = 999999999999
                    continue

                element_distances = []
                # for each element of in cluster I
                ci_length = len(all_clusters[i])
                for k in range(ci_length):
                    # for ci_length elements in cluster J
                    cj_length = len(all_clusters[j])
                    for l in range(cj_length):
                        # calculate distance between ci and cj
                        element_distances.append( euclideanDistance( all_clusters[i][k], all_clusters[j][l] ) )
                cluster_matrix[i][j] = min(element_distances)

        # find shortest cluster to cluster distance
        # print "cluster distance: " + str(cluster_distances)
        # print "cluster matrix: "
        # print cluster_matrix
        cluster_matrix = np.array(cluster_matrix)
        merge_pair = np.unravel_index(cluster_matrix.argmin(), cluster_matrix.shape)
        # merge I and best J into new array of clusters
        if length < 12:
            print "Merging: " + str(merge_pair)
            print "Distance: " + str(cluster_matrix.argmin())
        new_cluster = all_clusters[merge_pair[0]] + all_clusters[merge_pair[1]]
        all_clusters.append(new_cluster)
        if merge_pair[0] > merge_pair[1]:
            del all_clusters[merge_pair[0]]
            del all_clusters[merge_pair[1]]
        else:
            del all_clusters[merge_pair[1]]
            del all_clusters[merge_pair[0]]

        length = len(all_clusters)

def part_3_2(data):
    # Implement HAC algorithm using complete link to measure the distance
    # between clusters.

    all_clusters = []
    for d in data:
        all_clusters.append([d])

    length = len(all_clusters)

    #TODO
    while(length > 0):
        print " length: " + str(length)

        # for each cluster
        cluster_matrix = [[9999 for x in range(length)] for y in range(length)]
        for i in range(length):
            # TODO: implement skip list
            # compute best max distance from cluster to cluster
            cluster_distances = []
            # for each other cluster
            for j in range(length):
                # skip self
                if i == j:
                    cluster_matrix[i][j] = 999999999999
                    continue

                element_distances = []
                # for each element of in cluster I
                ci_length = len(all_clusters[i])
                for k in range(ci_length):
                    # for ci_length elements in cluster J
                    cj_length = len(all_clusters[j])
                    for l in range(cj_length):
                        # calculate distance between ci and cj
                        element_distances.append( euclideanDistance( all_clusters[i][k], all_clusters[j][l] ) )
                cluster_matrix[i][j] = max(element_distances)

        # find shortest cluster to cluster distance
        # print "cluster distance: " + str(cluster_distances)
        # print "cluster matrix: "
        # print cluster_matrix
        cluster_matrix = np.array(cluster_matrix)
        merge_pair = np.unravel_index(cluster_matrix.argmax(), cluster_matrix.shape)
        # merge I and best J into new array of clusters
        if length < 12:
            print "Merging: " + str(merge_pair)
            print "Distance: " + str(cluster_matrix.argmax())
        new_cluster = all_clusters[merge_pair[0]] + all_clusters[merge_pair[1]]
        all_clusters.append(new_cluster)
        if merge_pair[0] > merge_pair[1]:
            del all_clusters[merge_pair[0]]
            del all_clusters[merge_pair[1]]
        else:
            del all_clusters[merge_pair[1]]
            del all_clusters[merge_pair[0]]

        length = len(all_clusters)


def importing_data():
    print "Importing Data"
    data = 'data-2.txt'
    data = pd.read_csv(data, header=None)
    dfData = data.values[:,:]
    return dfData

if __name__ == '__main__':
    main()

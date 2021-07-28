import numpy as np
import math
import pandas as pd
import scipy.io
import random
import matplotlib.pyplot as plt

def get_coordinates_from_indices(coordinates, data):
    centroids = []
    # get co-ordinate points from indice values
    for i in coordinates:
        centroids.append(data.loc[i])
    return centroids

def get_initial_centroids(n, data):
    # get initial centroids by randomizing the indices
    initial_centroid_indices = random.sample(range(0, len(data)), n)
    centroids = get_coordinates_from_indices(initial_centroid_indices, data)
    return centroids

def compute_distance(x1, x2):
    # calculate distance between two points
    return sum((x1 - x2)**2)**0.5

def assign_to_centroid(centroids, data):
    centroid = []
    distance_sum = []
    # Assign the data point to the nearest centroid
    for i in data:
        distance = []
        for j in centroids:
            distance.append(compute_distance(i, j))
        # compute distance of point from the centroid for the objective function
        distance_sum.append(pow(distance[np.argmin(distance)] ,2))
        centroid.append(np.argmin(distance))
    return centroid, sum(distance_sum)

def compute_centroids(clstrs, data):
    centroids = []
    df = pd.concat([pd.DataFrame(data), pd.DataFrame(clstrs, columns=['cluster'])],axis=1)
    # calculate cluster centroids
    for c in set(df['cluster']):
        cur_clstr = df[df['cluster'] == c][df.columns[:-1]]
        clstr_mean = cur_clstr.mean(axis=0)
        centroids.append(clstr_mean)
    return centroids

def implement_k_means_strategy_1(samples):
    final_distance = []
    for j in range(2, 11):
        # compute initial centroids
        centroids_data = get_initial_centroids(j, samples)
        centroids_data = np.array(centroids_data)
        samples_array = np.array(samples)
        print("Initial centroids for k = {}".format(j))
        print(centroids_data)
        for i in range(10):
            # get the clustered data, distance of data points from clusters
            clustered_data, distance_sum = assign_to_centroid(centroids_data, samples_array)
            #compute centroids for the next iteration
            centroids_data = compute_centroids(clustered_data, samples_array)
        print("Final centroids for k = {}".format(j))
        print(np.array(centroids_data))
        # append final objective function value for each cluster value
        final_distance.append(distance_sum)
    print("Objective function values")
    print(np.array(final_distance))
    return final_distance

def draw_graph(y):
    # plot graph for objective function vs number of clusters
    x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(x, y)
    plt.title('elbow graph')
    plt.xlabel('No of Clusters')
    plt.ylabel('Objective Function')
    plt.show()

def get_other_centroids(centroids, n, data):
    # calculate centroids based on first centroid
    for i in range(1, n):
        new_centroids = []
        data = np.array(data)
        for sample in data:
            distance = 0
            for centroid in centroids:
                distance = distance + compute_distance(sample, centroid)
            # calculate average distance from the centroids
            avg_distance = distance / len(centroids)
            new_centroids.append(avg_distance)
        # get the index which gives max average distance
        max_dist_index = new_centroids.index(max(new_centroids))
        centroids.append(data[max_dist_index])
    return centroids


def get_initial_centroids_strategy_2(n, data):
    # randomly initialize first centroid indices
    initial_centroid_indices = random.sample(range(0, len(data)), 1)
    # get co-ordinates from the data for the centroid
    centroids = get_coordinates_from_indices(initial_centroid_indices, data)
    # get other centroid values using the first centroid
    centroids = get_other_centroids(centroids, n, data)
    return centroids

def implement_k_means_strategy_2(samples):
    final_distance = []
    for j in range(2, 11):
        #get initial centroids using strategy 2
        centroids_data = get_initial_centroids_strategy_2(j, samples)
        centroids_data = np.array(centroids_data)
        samples_array = np.array(samples)
        print("Initial centroids for k = {}".format(j))
        print(centroids_data)
        for i in range(20):
            # assign data point to closest centroid
            clustered_data, distance_sum = assign_to_centroid(centroids_data, samples_array)
            # compute centroids for next iteration
            centroids_data = compute_centroids(clustered_data, samples_array)
        print("Final centroids for k = {}".format(j))
        print(np.array(centroids_data))
        final_distance.append(distance_sum)
    print("Objective Function Values")
    print(final_distance)
    return final_distance

if __name__ == '__main__':
    data = scipy.io.loadmat('AllSamples.mat')
    samples = pd.DataFrame(data['AllSamples'])
    y1 = implement_k_means_strategy_1(samples)
    draw_graph(y1)
    y2 = implement_k_means_strategy_1(samples)
    draw_graph(y2)
    y3 = implement_k_means_strategy_2(samples)
    draw_graph(y3)
    y4 = implement_k_means_strategy_2(samples)
    draw_graph(y4)

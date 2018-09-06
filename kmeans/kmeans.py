import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import argparse

def print_it(it):
    print("#"*80)
    print("#" + " "*78 + "#")
    print("#\t\t\t\t  Iteration {}:\t\t\t\t       #".format(it))
    print("#" + " "*78 + "#")
    print("#"*80)

def load_data(data_path):
    """Return data from a txt file as array"""
    return np.loadtxt(data_path)

# TODO: function for data generation

def euclidian_dist(x, y):
    """Return the Euclidian distance between x and y (N dimentions)"""
    sum = 0
    #for each dimention of the points, sum the square diffrence
    for Xi, Yi in zip(x, y):
        sum += (Xi - Yi)*(Xi - Yi)
    return np.sqrt(sum)

def plot_data(data, save=False):
    """Simple plot of the data. /!\ 2 dims only for now"""

    plt.scatter(data[:,0], data[:,1])

    plt.title("Data visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    if save :
        plt.savefig("data_visualization.png")
    plt.show()


def plot_clusters(clusters, centroids, save=False):

    colors = iter(cm.rainbow(np.linspace(0, 1, len(centroids))))

    for clt, centroid in zip(clusters.keys(), centroids):
        color = next(colors)
        plt.scatter(clusters[clt][:,0], clusters[clt][:,1], c=color)
        plt.scatter(centroid[0],centroid[1], marker='x', c=color)

    plt.title("Clusters visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    if save :
        plt.savefig("clusters_visualization.png")
    plt.show()

# TODO: function for iteration visualization

def average_position_cluster(cluster):

    sums = [] #sum on each dimension
    nb_points = len(cluster)

    if nb_points == 0:
        return 0, 0

    #init the sums array with the good number of dims
    for dim in cluster[0]:
        sums.append(0)

    for point in cluster:
        for dim, value in enumerate(point):
            sums[dim] += value

    return np.asarray([(sums[0]/float(nb_points)) , (sums[1]/float(nb_points))])

def movement_rate(prev_centroids, centroids):

    sum_diff = 0
    for prev, centr in zip(prev_centroids, centroids):
        sum_diff += abs(prev[0] - centr[0])
        sum_diff += abs(prev[1] - centr[1])

    epsilon = sum_diff / float(len(centroids)) / float(len(centroids[0]))

    return epsilon

def random_centroids(nb_centroid, dims=2, min=0, max=1):
    centroids = []
    for i in range(nb_centroid):
        dim = []
        for j in range(dims):
            dim.append(random.uniform(min, max))
        centroids.append(dim)
    return np.asarray(centroids)

def mean_clusters_std(clusters):
    sum_std = 0
    for clt in clusters.keys():
        sum_std += np.std(clusters[clt])
    return sum_std/float(len(clusters))

def kmeans(data, k=2, iter=1, epsilon=0.001, verbose=None, distance='euclidian'):
    """This function use K-means algorithm to split the data into k clusters

    Args :
        iter: Number of times we will execute the algorithm from diffrent random points
        epsilon:

    """

    clusters, best_clusters = {}, {}
    best_centroids = []
    best_iter = -1
    best_std = sys.maxsize

    for it in range(iter):
        if verbose:
            print_it(it)

        centroids = random_centroids(k, data[0].shape[0],
                                     np.min(data), np.max(data))


        convergence = False
        cmpt = 0
        while not convergence and cmpt < 100:
            cmpt += 1

            prev_centroids = centroids.copy()

            # Reset clusters
            for i in range(k):
                clusters[i] = []

            #compute the distance between each point and eauch centroid
            for data_point in data:
                dist = []
                for centr_idx, centroid in enumerate(centroids):
                    dist.append(euclidian_dist(data_point, centroid))

                clusters[np.argmin(dist)].append(data_point)


            #Move centroides to the average of their points
            for clust_idx in clusters.keys():
                clusters[clust_idx] = np.asarray(clusters[clust_idx])
                centroids[clust_idx] = average_position_cluster(clusters[clust_idx])

            move_rate = movement_rate(prev_centroids, centroids)
            if verbose:
                print("Movement rate : {0:.5f}".format(move_rate))
            np.set_printoptions(precision=3)
            if verbose:
                print("From : \n{}\nTo : \n{}".format(prev_centroids, centroids))

            if(move_rate < epsilon):
                convergence = True

        # Compute std deviation and save the bests clusters
        standart_deviation = mean_clusters_std(clusters)
        if verbose:
            print("Standart deviation : {0:.5f}\n".format(standart_deviation))
        if(standart_deviation < best_std):
            best_std = standart_deviation
            best_clusters = clusters.copy()
            best_centroids = centroids.copy()
            best_iter = it

    if verbose:
        print("Best clusters found at iteration {}".format(best_iter))
    return best_clusters, best_centroids

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", help="increase output verbosity")
    args = parser.parse_args()

    # data = load_data("../data/data.txt")
    data = np.load("../data/clusterable_data.npy")

    clusters, centroids = kmeans(data, k=6, iter=2, verbose=args.verbosity)
    plot_data(data)
    plot_clusters(clusters, centroids)

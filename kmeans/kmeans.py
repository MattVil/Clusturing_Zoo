import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    x, y = [], []
    for point in data:
        x.append(point[0])
        y.append(point[1])

    plt.scatter(x, y)

    plt.title("Data visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    if save :
        plt.savefig("data_visualization.png")
    plt.show()


def plot_clusters(clusters, centroids, save=False):

    xCentroid, yCentroid = [], []
    color = ['b', 'g', 'y', 'o']
    for centroid in centroids:
        xCentroid.append(centroid[0])
        yCentroid.append(centroid[1])
    plt.scatter(xCentroid, yCentroid, c='r')

    for clt in clusters.keys():
        x, y = [], []
        for point in clusters[clt]:
            x.append(point[0])
            y.append(point[1])
        plt.scatter(x, y, c=color[clt])

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

    #init the sums array with the good number of dims
    for dim in cluster[0]:
        sums.append(0)

    for point in cluster:
        for dim, value in enumerate(point):
            sums[dim] += value

    return (sums[0]/float(nb_points) , (sums[1]/float(nb_points)))

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
            dim.append(random.random()*max+min)
        centroids.append(dim)
    return centroids


def kmeans(data, k=2, iter=1, epsilon=0.1, distance='euclidian'):
    """This function use K-means algorithm to split the data into k clusters

    Args :
        iter: Number of times we will execute the algorithm from diffrent random points
        epsilon:

    """

    clusters = {}

    for it in range(iter):
        print_it(it)

        centroids = random_centroids(2, 2, 0, 2)


        convergence = False

        while not convergence:

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
                centroids[clust_idx] = average_position_cluster(clusters[clust_idx])

            move_rate = movement_rate(prev_centroids, centroids)
            print("Movement rate : {}".format(move_rate))
            print("From\t{} \nTo\t{}\n".format(prev_centroids, centroids))

            if(move_rate < epsilon):
                convergence = True

        # TODO: compute std deviation and save the bests clusters

    return clusters, centroids

if __name__ == '__main__':
    data = load_data("./data.txt")
    clusters, centroids = kmeans(data)
    #plot_data(data)
    plot_clusters(clusters, centroids, True)

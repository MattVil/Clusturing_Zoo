import numpy as np
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

def kmeans(data, k=2, iter=10, epsilon=None, distance='euclidian'):

    clusters = {}
    centroids = [(0.15, 0.1), (0.5, 0.1)]

    # TODO: complexe choise op
    for it in range(iter):
        print_it(it)

        print("Centroids before move : \t{}".format(centroids))

        # Reset clusters
        for i in range(k):
            clusters[i] = []

        #compute the distance between each point and eauch centroid
        for data_point in data:
            dist = []
            for centr_idx, centroid in enumerate(centroids):
                dist.append(euclidian_dist(data_point, centroid))

            clusters[np.argmin(dist)].append(data_point)
            #print(dist)
        #print(clusters[0])
        #print(clusters[1])

        #Move centroides to the average of their points
        for clust_idx in clusters.keys():
            centroids[clust_idx] = average_position_cluster(clusters[clust_idx])

        print("Centroids after move : \t{}".format(centroids))

    return clusters, centroids

if __name__ == '__main__':
    data = load_data("./data.txt")
    clusters, centroids = kmeans(data)
    #plot_data(data)
    plot_clusters(clusters, centroids, True)

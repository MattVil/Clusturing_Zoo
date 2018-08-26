import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def average_position_cluster(cluster):

    sums = []
    nb_points = len(data)
    print(cluster[0])
    for dim in range(len(cluster[0])):
        sums.append(0)

    for point in cluster:
        for dim, value in enumerate(point):
            sums[dim] += value

    return (sums[0]/float(nb_points) , (sums[1]/float(nb_points)))

def kmeans(data, k=2, iter=10, epsilon=None, distance='euclidian'):

    clusters = {}
    centroids = [(0, 0), (1, 1)]

    # TODO: complexe choise op
    for it in range(iter):

        print("Centroids before move : \t{}".format(centroids))

        # Reset clusters
        for i in range(k):
            clusters[i] = []

        #compute the distance between each point and eauch centroid
        for data_point in data:
            min_dist = 100000
            for centr_idx, centroid in enumerate(centroids):
                dist = euclidian_dist(data_point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_centr = centr_idx
            clusters[closest_centr].append(data_point)
        print(clusters[0])
        print(clusters[1])
        #Move centroides to the average of their points
        for clust_idx in clusters.keys():
            centroids[clust_idx] = average_position_cluster(clusters[clust_idx])

        print("Centroids after move : \t{}".format(centroids))

if __name__ == '__main__':
    data = load_data("./data.txt")
    kmeans(data)
    plot_data(data)

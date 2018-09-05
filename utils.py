import numpy as np
import matplotlib.pyplot as plt

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

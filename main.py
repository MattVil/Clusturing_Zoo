from __future__ import absolute_import

import argparse
import numpy as np
import kmeans as km

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", help="increase output verbosity")
    args = parser.parse_args()

    # data = load_data("../data/data.txt")
    data = np.load("data/clusterable_data.npy")

    clusters, centroids = km.kmeansClustering(data, k=6, iter=20, verbose=args.verbosity)
    plot_data(data)
    plot_clusters(clusters, centroids)

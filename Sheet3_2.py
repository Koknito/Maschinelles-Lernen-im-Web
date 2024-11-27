import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering


def checkerboard():
    # Gaussian mixture checkerboard
    # 16 groups
    size = 300
    x = np.array([[0, 0]])
    y = np.array([])
    for i in range(0, 4):
        for j in range(0, 4):
            group = np.random.multivariate_normal([i * 8, j * 8], [[1, 0], [0, 3]], size)
            x = np.concatenate((x, group))
            y = np.concatenate((y, np.repeat(i * 4 + j, size, axis=0)))
            plt.scatter(group[:, 0], group[:, 1])
    x = x[1:]
    plt.title("original - checkerboard")
    plt.show()
    return x, y


def circle():
    # 10 groups
    size = 300
    x = np.array([[0, 0]])
    y = np.array([])
    for i in range(10):
        group = []
        dists = np.random.multivariate_normal([i * 6], [[1]], size)[:, 0]
        angles = np.random.uniform(0, 2 * np.pi, size)
        for j in range(size):
            group.append([np.cos(angles[j]) * dists[j], np.sin(angles[j]) * dists[j]])
        group = np.array(group)
        x = np.concatenate((x, group))
        y = np.concatenate((y, np.repeat(i, size, axis=0)))
        plt.scatter(group[:, 0], group[:, 1])
    x = x[1:]
    plt.title("original - circle")
    plt.show()
    return x, y


def calc_sim(clusters, y_found, y, name):
    assert (np.max(y_found) + 1 >= clusters)
    conf_mat = metrics.confusion_matrix(y, y_found)
    identical = 0
    for i in range(conf_mat.shape[0]):
        identical += np.max(conf_mat[:, i])
    print(name + "(" + str(np.max(y_found) + 1) + " clusters) accuracy:" + str(identical / (clusters * 300)))


def circle_to_line(x):
    result = []
    for i in x:
        result.append([np.sqrt(i[0] ** 2 + i[1] ** 2)])
    return np.array(result)


def kmeans(x_draw, y, x, clusters, name):
    x_kmeans = KMeans(n_clusters=clusters, random_state=0).fit(x).predict(x)
    for i in range(np.max(x_kmeans) + 1):
        plt.scatter(x_draw[x_kmeans == i, 0], x_draw[x_kmeans == i, 1])
    plt.title("KMeans - " + name + "(" + str(np.max(x_kmeans) + 1) + " classes)")
    plt.show()
    calc_sim(clusters, x_kmeans, y, "KMeans - " + name)


def affinity(x_draw, y, x, clusters, damping, max_iter, name):
    x_affinity = AffinityPropagation(damping=damping, max_iter=max_iter, random_state=0).fit(x).predict(x)
    for i in range(np.max(x_affinity) + 1):
        plt.scatter(x_draw[x_affinity == i, 0], x_draw[x_affinity == i, 1])
    plt.title("Affinity propagation - " + name + "(" + str(np.max(x_affinity) + 1) + " classes)")
    plt.show()
    calc_sim(clusters, x_affinity, y, "Affinity propagation - " + name)


def spectral(x_draw, y, x, clusters, neighbors, name):
    x_spectral = SpectralClustering(n_clusters=clusters, n_neighbors=neighbors, n_init=10, random_state=0).fit_predict(
        x)
    for i in range(np.max(x_spectral) + 1):
        plt.scatter(x_draw[x_spectral == i, 0], x_draw[x_spectral == i, 1])
    plt.title("Spectral clustering - " + name + "(" + str(np.max(x_spectral) + 1) + " classes)")
    plt.show()
    calc_sim(clusters, x_spectral, y, "Spectral clustering - " + name)


x_check, y_check = checkerboard()
x_circ, y_circ = circle()
x_circ_as_line = circle_to_line(x_circ)

kmeans(x_check, y_check, x_check, 16, "checkerboard")
kmeans(x_circ, y_circ, x_circ_as_line, 10, "circle")
affinity(x_check, y_check, x_check, 16, 0.95, 200, "checkerboard")
affinity(x_circ, y_circ, x_circ_as_line, 10, 0.8, 200, "circle")
spectral(x_check, y_check, x_check, 16, 3, "checkerboard")
spectral(x_circ, y_circ, x_circ_as_line, 10, 3, "circle")

# Schachbrett
# 1 - Spectral clustering: Macht zu Vorbereitung mehrere KMeans(gut für das Problem), 3 ist gut gewählt
# 2 - Affinity propagation: Nutzt etwas zu viele Gruppen, sonst gut
# 3 - KMeans: Lachhaft, aber passend da er in lokalen Optima gefangen bleibt

# Kreis
# 1 - Spectral clustering: Allgemein haben es alle sehr gut gemacht, mehrere KMeans
# 2/3 - KMeans/Affinity propagation: Allgemein haben es alle sehr gut gemacht, mehrere KMeans


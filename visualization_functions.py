import numpy as np
import matplotlib.pyplot as plt

COLORS = ['lightcoral', 'mediumturquoise', 'yellowgreen', 'mediumpurple', 'navajowhite', 'lightgreen', 'tan', 'pink',
          'lightgrey', 'teal', 'blue']


def scale_node_sizes(array: np.ndarray, x0=0, y0=16, x1=2, y1=8, xn=10, yn=0.1) -> np.ndarray:
    def lagrange_polynomial(x):
        part1 = y0 * ((x - x1) * (x - xn)) / ((x0 - x1) * (x0 - xn))
        part2 = y1 * ((x - x0) * (x - xn)) / ((x1 - x0) * (x1 - xn))
        part3 = yn * ((x - x0) * (x - x1)) / ((xn - x0) * (xn - x1))
        return part1 + part2 + part3
    return np.clip(np.exp(1 / (array + 0.1)) + lagrange_polynomial(array), 0.025, 500)


def colorize_nodes_by_layer(layers: np.ndarray):
    layer_to_color = dict(enumerate(COLORS))
    return [layer_to_color.get(layer, "blue") for layer in layers]


def layer_l2_scatter_plot(layers_modified: np.ndarray, layers_common: np.ndarray, pos_modified: np.ndarray, pos_common: np.ndarray):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    l2 = np.sqrt(np.sum(pos_modified ** 2, axis=1))
    pearson_correlation = np.corrcoef(np.stack((layers_modified, l2), axis=0))[0, 1]
    print(pearson_correlation)
    plt.scatter(layers_modified, l2, label='узел')
    plt.title('Модифицированный силовой алгоритм')
    plt.xlabel('Радиус от центрального узла')
    plt.ylabel('Длина вектора')
    plt.legend()

    plt.subplot(1, 2, 2)
    l2 = np.sqrt(np.sum(pos_common ** 2, axis=1))
    pearson_correlation = np.corrcoef(np.stack((layers_common, l2), axis=0))[0, 1]
    print(pearson_correlation)
    plt.scatter(layers_common, l2, label='узел')
    plt.title('Fruchterman and Reingold')
    plt.xlabel('Радиус от центрального узла')
    plt.ylabel('Длина вектора')
    plt.legend()

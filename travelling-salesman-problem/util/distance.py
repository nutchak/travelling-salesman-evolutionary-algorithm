import numpy as np


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2).round()


def calculate_city_distance(distance_matrix, city_order):
    total_distances = distance_matrix[city_order[-1]][city_order[0]] \
                      + sum([distance_matrix[city_order[i]][city_order[i + 1]]
                             for i in range(len(city_order) - 1)])
    return total_distances


if __name__ == '__main__':
    point1 = np.array((6734, 1453))
    point2 = np.array((2233, 10))
    print(np.linalg.norm(point1 - point2).round())


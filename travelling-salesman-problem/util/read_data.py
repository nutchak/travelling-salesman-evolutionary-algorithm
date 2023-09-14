import numpy as np
from scipy.spatial.distance import pdist, squareform


def read_txt_coord(file_name):
    city_coord = squareform(pdist(np.loadtxt(file_name)))
    return city_coord


def read_txt_distance_matrix(file_name):
    city_coord = np.loadtxt(file_name)
    return city_coord

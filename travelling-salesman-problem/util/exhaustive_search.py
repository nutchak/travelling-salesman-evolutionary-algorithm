import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import inf
from tqdm import tqdm

from distance import calculate_city_distance
from read_data import read_txt_distance_matrix


def city_permutations(count: int) -> np.ndarray:
    """All possible permutations of cities

    Parameters
    ----------
    count : `int`
        Number of cities

    Returns
    -------
    `np.ndarray`
        Permutations of cities
    """
    # itertools.permutations for creating all possible permutations
    # The result is saved as an array of arrays of city permutations
    return np.asarray(list(itertools.permutations(range(count))))


def exhaustive_search_shortest(distance_matrix: np.ndarray, count: int) -> tuple:
    """Exhaustive search for given number of cities

    Parameters
    ----------
    distance_matrix: `np.ndarray`
        Matrix of city distances
    count : `int`
        Number of cities

    Returns
    -------
    `Tuple`
        A tuple of count, city order, minimum distance, and time taken
    """
    # Recording start time
    # Just in case
    start = time.thread_time()

    # All possible permutations of given number of cities
    perm = city_permutations(count)

    # Order of cities that has minimum distance
    cities_order = []

    # Minimum distance
    min_distance = inf

    # tqdm is also used to check the time taken (also for each iteration)
    for i in tqdm(range(len(perm) - 1)):
        permutation_distance = calculate_city_distance(distance_matrix, perm[i])
        if permutation_distance < min_distance:
            min_distance = permutation_distance
            cities_order = perm[i]

    # Recording stop time
    stop = time.thread_time()
    return count, cities_order, min_distance, stop-start


def exhaustive_search_result(distance_matrix: np.ndarray, start: int, stop: int) -> str:
    """Calculate the results of exhaustive search from start to stop city

    Parameters
    ----------
    distance_matrix: `np.ndarray`
        Matrix of city distances
    start : int
        Number of start city
    stop : int
        Number of stop city

    Returns
    -------
    df : `pd.Dataframe`
        The results (Number of city, City order, Minimum distance, Time) from dataframe are saved into .csv
    """
    # Create a dataframe with columns listed below
    df = pd.DataFrame(columns=["Number of city", "City order", "Minimum distance", "Time"])
    for i in range(start, stop+1):
        # Add the result to dataframe
        df.loc[len(df.index)] = exhaustive_search_shortest(distance_matrix, i)
    # Plot for time taken for numbers of cities
    # df.plot(x="Number of city", y="Time", kind="line", subplots=True, title="Time taken", xlabel="Number of city",
    #         ylabel="Time", legend=True, label="Time(s)")
    # plt.grid()
    # plt.show()
    # Save the dataframe to csv
    df.to_csv("exhaustive_search_results.csv", index=False)
    return f"6 cities: {df.loc[0].at['City order']} \n" \
           f"10 cities: {df.loc[4].at['City order']}"


if __name__ == "__main__":
    # Exhaustive search from 6 to 10 cities
    # Return plots of time taken, plot of 6 cities, and plot of 10 cities
    print(exhaustive_search_result(distance_matrix=read_txt_distance_matrix("att48/att48_d.txt"), start=6, stop=10))


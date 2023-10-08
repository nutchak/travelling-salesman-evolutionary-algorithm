import numpy as np
import pandas as pd
from tqdm import tqdm

from distance import calculate_city_distance


def hill_climbing(distance_matrix, iteration: int, count: int, runs: int) -> pd.DataFrame:
    """Hill climbing for finding possible solutions for tsp by swapping randomized positions

    Parameters
    ----------
    distance_matrix: `np.ndarray`
        Matrix of city distances
    iteration: `int`
        Number of iteration
    count : `int`
        Number of cities
    runs : `int`
        Number of runs

    Returns
    -------
    df : `pd.Dataframe`
        Dataframe of city orders and their distances
    """
    assert count > 1

    df = pd.DataFrame(columns=["City order", "Distance"])

    for _ in tqdm(range(runs)):
        # Randomize permutation
        city_order = np.asarray(np.random.permutation(count))
        min_distance = calculate_city_distance(distance_matrix, city_order)

        for _ in range(iteration):
            # Get two indexes for swapping
            city1 = np.random.randint(low=0, high=count - 1)
            city2 = np.random.randint(low=0, high=count - 1)
            while city2 == city1:
                city2 = np.random.randint(low=0, high=count - 1)

            current_solution = city_order.copy()
            # Swap the alleles at indexes above
            current_solution[city1], current_solution[city2] = current_solution[city2], current_solution[city1]
            # Calculate distance of the new solution
            current_distance = calculate_city_distance(distance_matrix, current_solution)

            if current_distance < min_distance:
                min_distance = current_distance
                city_order = current_solution

        df.loc[len(df.index)] = city_order, min_distance
    df.to_csv("hill_climbing_results.csv", index=False)
    return df


def hill_climbing_result(iteration: int, count: int, runs: int) -> str:
    """Show the result of hill climbing in text format

    Parameters
    ----------
    iteration : `int`
        Number of iterations
    count : `int`
        Number of cities
    runs : `int`
        Number of runs
    Returns
    -------
    str
    """
    df = hill_climbing(iteration, count, runs)
    return f"Minimum cities order: {df['City order'].loc[df['Distance'].idxmin()]} \n" \
           f"Minimum distance(km): {df['Distance'].min():.2f} \n" \
           f"Maximum cities order: {df['City order'].loc[df['Distance'].idxmax()]} \n" \
           f"Maximum distance(km): {df['Distance'].max():.2f} \n" \
           f"Mean(km): {df['Distance'].mean():.2f} \n" \
           f"Standard deviation(km): {df['Distance'].std():.2f} \n"


import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from util.read_data import read_txt_distance_matrix
from util.distance import calculate_city_distance
from util.evolutionary_utils import generate_population, linearly_ranking, sus, recombination, mutation


def evolutionary_algorithm(
        population_count: int,
        city_count: int,
        generation: int,
        s_parameter: float,
        distance_matrix: np.ndarray):
    """
    Evolutionary algorithm to find solutions for tsp problem
    :param population_count: A number of initial population
    :param city_count: A number of cities
    :param generation: A number of generation
    :param s_parameter: A parameter for Stochastic universal sampling
    :param distance_matrix: A numpy array of distance matrix
    :return:
    """
    start = time.thread_time()
    # Create dataframe for results of each generation
    best_result = pd.DataFrame(
        columns=["Generation",
                 "City order",
                 "Minimum distance",
                 "Maximum distance",
                 "Average distance",
                 "Standard deviation"])
    # Generate initial population
    # Columns = City order, Distance
    population = generate_population(population_count, city_count, distance_matrix)
    # Evaluate fitness
    # Population(ranked) with columns -> Probability, Cumulative probability
    population = linearly_ranking(population, s_parameter)

    for i in tqdm(range(generation)):
        # Choose 50% of population for recombination
        parents = population.loc[sus(population, int(population_count / 2))]
        # Create new population by recombination
        # Create 5 * population_count
        offsprings = recombination(parents, (population_count * 5))
        # Add offsprings to population
        for offspring in offsprings:
            population.loc[len(population.index)] = offspring, \
                                                    calculate_city_distance(
                                                        distance_matrix,
                                                        offspring), np.nan, np.nan
        # Mutation
        population = mutation(population)
        # Evaluate new population
        population = linearly_ranking(population, s_parameter)
        # Add the results to the result dataframe
        best_result.loc[len(best_result.index)] = \
            i + 1, \
            population.iloc[len(population.index) - 1]["City order"], \
            population.iloc[len(population.index) - 1]["Distance"], \
            population["Distance"].max(), \
            population["Distance"].mean(), \
            population["Distance"].std()
        # Choose top 10% individuals
        top = population[int((len(population.index) * 90) / 100):]
        # Randomly choosing 100 individuals from the population
        population_100 = population.loc[sus(population, population_count)]
        # Concatenate the population
        population = pd.concat([top, population_100])
        population = linearly_ranking(population, s_parameter)
    stop = time.thread_time()
    return best_result, stop - start


def genetic_algorithm_results(
        populations_count: int,
        city_count: int,
        generation_count: int,
        runs: int,
        distance_matrix: np.ndarray):
    """
    Best result from each run
    :param populations_count: A number of population
    :param city_count: A number of cities
    :param generation_count: A number of generation
    :param runs: A number of runs
    :param distance_matrix: A numpy array of distance matrix
    :return:
    """
    df_best_result_all = pd.DataFrame(
        columns=["Generation",
                 "City order",
                 "Minimum distance",
                 "Maximum distance",
                 "Average distance",
                 "Standard deviation"])
    df = pd.DataFrame(
        columns=["Number of run",
                 "Best generation",
                 "City order",
                 "Minimum distance",
                 "Maximum distance",
                 "Average distance",
                 "Standard deviation",
                 "Total time"])
    for i in range(runs):
        best_result = evolutionary_algorithm(populations_count,
                                             city_count,
                                             generation_count,
                                             s_parameter=1.5,
                                             distance_matrix=read_txt_distance_matrix("att48/att48_d.txt"))
        df_best_result_all = pd.concat([df_best_result_all, best_result[0]], ignore_index=True)
        df.loc[len(df.index)] = \
            i + 1, \
            best_result[0]["Minimum distance"].idxmin() + 1, \
            best_result[0]["City order"].iat[len(best_result[0]) - 1], \
            best_result[0]["Minimum distance"].iat[len(best_result[0]) - 1], \
            best_result[0]["Maximum distance"].iat[len(best_result[0]) - 1], \
            best_result[0]["Average distance"].iat[len(best_result[0]) - 1], \
            best_result[0]["Standard deviation"].iat[len(best_result[0]) - 1], \
            best_result[1]
    # # Save values for plotting
    # x = [df_best_result_all["Generation"].iat[i] for i in range(generation_count)]
    # y = [df_best_result_all.where(df_best_result_all["Generation"] == i + 1)["Minimum distance"].mean() for i in
    #      range(generation_count)]
    # np.save(f"variation_plot/pop{populations_count}_x.npy", x)
    # np.save(f"variation_plot/pop{populations_count}_y.npy", y)
    # df_best_result_all.to_csv(
    #     f"out_genetic/best_result/all_best_result_{populations_count}_{city_count}_{generation_count}_{runs}.csv",
    #     index=False)
    # df.to_csv(
    #     f"out_genetic/whole_generation/whole_generation_{populations_count}_{city_count}_{generation_count}_{runs}.csv",
    #     index=False)
    # plot_plan(get_city(df['City order'].loc[df['Minimum distance'].idxmin()]))
    print(f"Minimum distance(km): {df['Minimum distance'].min()}")
    print(f"Maximum distance(km): {df['Maximum distance'].max()}")
    print(f"Average distance(km): {df['Average distance'].mean()}")
    print(f"Standard deviation(km): {df['Standard deviation'].std()}")
    return df, df_best_result_all


if __name__ == '__main__':
    # Over 3 population size
    for pop_count in [25, 50, 100]:
        print(genetic_algorithm_results(populations_count=pop_count,
                                        city_count=48,
                                        generation_count=100,
                                        runs=5,
                                        distance_matrix=read_txt_distance_matrix("att48/att48_d.txt")))

import numpy as np
import pandas as pd
from distance import calculate_city_distance


MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.90
rng_permutation = np.random.RandomState(seed=42)


def generate_population(population_count: int, city_count: int, distance_matrix: np.ndarray) -> pd.DataFrame:
    """
    Generate initial population, calculate distance, and put in dataframe
    :param population_count: The number of initial population
    :param city_count: The number of cities
    :param distance_matrix: A numpy array of distance matrix
    :return: A dataframe consists of permutation of city orders and calculated distances
    """
    population = pd.DataFrame(columns=["City order", "Distance"])
    for i in range(population_count):
        # Use seed for generating permutations of cities
        chromosome = rng_permutation.permutation(city_count)
        population.loc[len(population.index)] = chromosome, calculate_city_distance(distance_matrix=distance_matrix,
                                                                                    city_order=chromosome)
    return population


def rank_population(population: pd.DataFrame) -> pd.DataFrame:
    """
    Rank the population based on distances
    :param population: A unranked dataframe
    :return: A ranked dataframe based on distances
    """
    # Rank based on distance
    # Worst to best
    population["Rank"] = population["Distance"].rank(method="first", ascending=False).astype(int) - 1
    population = population.set_index(keys="Rank", drop=True)
    population = population.sort_values("Rank")
    return population


def linearly_ranking(population: pd.DataFrame, s: float) -> pd.DataFrame:
    """
    Ranking the population linearly (0 is the worst), and calculate probability and cumulative probability
    based on their ranks
    :param population: A dataframe of population
    :param s: A parameter for stochastic universal sampling
    :return: A linearly ranked dataframe of population
    """
    population_count = len(population.index)
    population = rank_population(population)
    # Probability based on ranks
    # Use the calculation from "Introduction to Evolutionary Computing" chapter 5 page 82
    # P(lin_rank_i) = (2-s)/mu + 2i(s-1)/mu(mu-1)
    population["Probability"] = [((2 - s) / population_count)
                                 + (2 * i * (s - 1)
                                    / (population_count * (population_count - 1))) for i in
                                 range(len(population.index))]
    population["Cumulative probability"] = population["Probability"].cumsum()
    return population


def fps_selection(population: pd.DataFrame, count: int) -> pd.DataFrame:
    """
    Select parents for combination
    :param population: A dataframe of all population
    :param count: A number of selected parents
    :return: A dataframe of parents
    """
    parents = population.sample(n=count, weights=population["Probability"])
    return parents


def sus(population: pd.DataFrame, count: int):
    """
    Stochastic universal sampling
    :param population: A dataframe of population
    :param count: A number of chosen population
    :return:
    """
    distance_pointer = 1 / count
    start = np.random.uniform(low=0, high=distance_pointer)
    pointers = [start + (distance_pointer * i) for i in range(count)]
    return rws(population, pointers)


def rws(population: pd.DataFrame, pointers: list) -> list:
    """
    Stochastic universal sampling
    :param population: A dataframe of population
    :param pointers: A list of pointers (from Stochastic universal sampling)
    :return: List of indexes of chosen populations
    """
    keep = []
    for p in pointers:
        i = 0
        while population["Cumulative probability"].at[i] < p:
            i += 1
        keep.append(i)
    return keep


def inversion_mutation(chromosome: np.ndarray) -> np.ndarray:
    """
    Inversion mutation of a chromosome
    :param chromosome: Chromosome to mutate
    :return: Mutated chromosome
    """
    mutated_chromosome = chromosome.copy()
    i = np.random.randint(low=1, high=len(chromosome) - 3)
    j = np.random.randint(low=i + 1, high=len(chromosome) - 2)

    assert i < j

    mutated_chromosome[i:j] = mutated_chromosome[i:j][::-1]
    return mutated_chromosome


def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Partially mapped crossover of parents
    :param parent1: A permutation of parent 1
    :param parent2: A permutation of parent 2
    :return: A numpy array of two resulted children
    """
    # Initialize children, filled with number that is more than number of cities
    child1 = np.full(len(parent2), fill_value=100)
    child2 = np.full(len(parent1), fill_value=100)
    # Starting from 1 to len(parent1) -3 (so there would be three segments)
    i = np.random.randint(low=1, high=len(parent1) - 3)
    j = np.random.randint(i + 1, len(parent1) - 2)

    # Replace the middle segments for both children
    child1[i:j] = parent1[i:j]
    child2[i:j] = parent2[i:j]

    # pmx helper function
    child1 = pmx(parent2, child1, i, j)
    child2 = pmx(parent1, child2, i, j)

    return np.array([child1.astype(int), child2.astype(int)])


def pmx(parent: np.ndarray, child: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Helper function for pmx_crossover
    :param parent: A permutation of parent
    :param child: A permutation of child
    :param i:
    :param j:
    :return: A numpy array of mutated child
    """
    for index, x in enumerate(parent[i:j]):
        index += i
        # Check elements from parent's segment not in child
        while x not in child:
            element_in_child = child[index]
            if element_in_child not in parent[i:j]:
                parent_index_to_replace = np.where(parent == element_in_child)[0][0]
                child[parent_index_to_replace] = x
            else:
                index = np.where(parent == element_in_child)[0][0]

    # Fill in the rest
    for index, x in enumerate(child):
        if child[index] == 100:
            child[index] = parent[index]
    return child


def recombination(parents: pd.DataFrame, population_count: int) -> list:
    """
    Recombination of two parents
    :param parents: A dataframe of parents
    :param population_count: A number of targeted population
    :return: A list of offsprings
    """
    # Choose 10 parents for the mating pool
    offsprings = []
    i = len(offsprings)
    while i < population_count:
        chosen_parents = parents.sample(n=2)
        r = np.random.uniform(low=0, high=1)
        if r < CROSSOVER_RATE:
            # Choose 2 parents for crossover
            children = pmx_crossover(chosen_parents["City order"].iat[0], chosen_parents["City order"].iat[1])
            # print(children)
            for child in children:
                offsprings.append(child)
                i += 1
    return offsprings


def mutation(population: pd.DataFrame) -> pd.DataFrame:
    """
    Mutation of population
    :param population: A dataframe of population
    :return: A dataframe of mutated population
    """
    for i in range(len(population.index)):
        r = np.random.uniform(low=0, high=1)
        if r < MUTATION_RATE:
            # print(i)
            # print(population["City order"].iat[i])
            mutated_chromosome = inversion_mutation(population["City order"].iat[i])
            # print(mutated_chromosome)
            population["City order"].iat[i] = mutated_chromosome
            # print(population["City order"].iat[i])
    return population

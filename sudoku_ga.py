import numpy as np
import random


# Fitness function: counts the number of valid rows, columns, and subgrids
def fitness_func(sudoku):
    score = 0

    # Check rows and count valid entries
    for row in sudoku:
        score += len(set(row))  # number of unique numbers in the row

    # Check columns
    for col in range(9):
        column = [sudoku[row][col] for row in range(9)]
        score += len(set(column))  # number of unique numbers in the column

    # Check 3x3 subgrids
    for i in range(3):
        for j in range(3):
            subgrid = [sudoku[x][y] for x in range(i * 3, (i + 1) * 3) for y in range(j * 3, (j + 1) * 3)]
            score += len(set(subgrid))  # number of unique numbers in the subgrid

    return score


# Crossover: combines two Sudoku grids by swapping rows or subgrids
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 8)
    child = np.copy(parent1)

    # Swap rows
    child[crossover_point:] = parent2[crossover_point:]

    return child


# Mutation: randomly swaps values in a row or column
def mutation(sudoku, mutation_rate):
    if random.random() < mutation_rate:
        row1, row2 = random.sample(range(9), 2)
        col1, col2 = random.sample(range(9), 2)

        # Swap two cells in the grid
        sudoku[row1][col1], sudoku[row2][col2] = sudoku[row2][col2], sudoku[row1][col1]

    return sudoku


# Generates an initial population of random Sudoku grids
def generate_initial_pop(pop_size, initial_sudoku=None):
    population = []

    if initial_sudoku is not None:
        population.append(initial_sudoku)

    for _ in range(pop_size - 1):
        sudoku = np.random.randint(1, 10, (9, 9))  # Random grid with numbers 1-9
        population.append(sudoku)

    return population


# Tournament selection: selects the best individuals from a random subset
def tournament_selection(population, fitness_scores, tournament_size=6):
    selected_parents = []

    for _ in range(len(population) // 2):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]  # Select the individual with the highest fitness
        selected_parents.append(winner)

    return selected_parents


# Elitism: preserves the best individuals in the population
def elitism(population, fitness_scores, elite_size):
    elite = [population[i] for i in
             sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]]
    return elite


# Genetic algorithm to solve Sudoku using GA
def genetic_alg_sudoku(initial_sudoku, pop_size, generations, mutation_rate, elite_size,
                       max_no_improvement):
    population = generate_initial_pop(pop_size, initial_sudoku)
    best_fitness = -1
    no_improvement = 0

    for generation in range(generations):
        fitness_scores = [fitness_func(individual) for individual in population]
        best_gen_fitness = max(fitness_scores)

        # Print fitness score for the current generation
        print(f"Generation {generation}: Best Fitness = {best_gen_fitness}")

        # Stop early if solution is found
        if best_gen_fitness == 243:  # 9 rows + 9 columns + 9 subgrids
            print(f"Solution found at generation {generation}")
            solution = population[fitness_scores.index(best_gen_fitness)]
            print(solution)
            break

        if best_gen_fitness == best_fitness:
            no_improvement += 1
        else:
            no_improvement = 0

        if no_improvement >= max_no_improvement:
            print(f"Stopping early at generation {generation} due to stagnation.")
            print(f"Best fitness at stagnation: {best_gen_fitness}")
            break

        # Keep track of best fitness
        best_fitness = best_gen_fitness

        # Elitism: Keep best individuals
        elite = elitism(population, fitness_scores, elite_size)

        # Select parents for crossover
        selected_parents = tournament_selection(population, fitness_scores)

        # Generate next population
        new_population = elite[:]
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return population


# Main function to test the algorithm
def main():
    # Test case 1
    """initial_sudoku = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])"""

    #Test case 2
    """initial_sudoku = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])"""
    # Test case 3
    """initial_sudoku = np.array ([
        [5, 3, 3, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 2, 0, 0, 0, 6, 0],
        [8, 0, 0, 7, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])"""
    # Test case 4 - easy sudoku
    """initial_sudoku = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])"""
    # Test Case 5 - easy example sudoku - mostly completed
    initial_sudoku = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    print("Starting Sudoku Solver with Genetic Algorithm...")
    genetic_alg_sudoku(initial_sudoku=initial_sudoku, pop_size=250, generations=5000, mutation_rate=0.2, elite_size=40,
                       max_no_improvement=100)


if __name__ == "__main__":
    main()

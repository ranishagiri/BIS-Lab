import numpy as np
import random
import matplotlib.pyplot as plt


def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1.0


population_size = 100
num_generations = 50
mutation_rate = 0.01
crossover_rate = 0.7
chromosome_length = 16  

def initialize_population():
    population = []
    for _ in range(population_size):
        chromosome = ''.join(random.choice(['0', '1']) for _ in range(chromosome_length))
        population.append(chromosome)
    return population


def decode(chromosome):
    integer_value = int(chromosome, 2)
    x = integer_value / (2**chromosome_length - 1)
    return x


def evaluate_fitness(population):
    fitness_values = []
    for chromosome in population:
        x = decode(chromosome)
        fitness = fitness_function(x)
        fitness_values.append(fitness)
    return fitness_values

# Function to perform roulette wheel selection
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    selected_indices = np.random.choice(
        range(population_size), size=population_size, p=selection_probs
    )
    selected_population = [population[i] for i in selected_indices]
    return selected_population

# Function to perform crossover between two parents
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, chromosome_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Function to perform mutation on a chromosome
def mutate(chromosome):
    mutated_chromosome = ''
    for gene in chromosome:
        if random.random() < mutation_rate:
            mutated_gene = '1' if gene == '0' else '0'
        else:
            mutated_gene = gene
        mutated_chromosome += mutated_gene
    return mutated_chromosome

# Main GA loop
population = initialize_population()
best_fitness_history = []
best_individual_history = []

for generation in range(num_generations):
    fitness_values = evaluate_fitness(population)
    best_fitness = max(fitness_values)
    best_individual = population[fitness_values.index(best_fitness)]
    best_fitness_history.append(best_fitness)
    best_individual_history.append(decode(best_individual))
    print(f'Generation {generation}: Best Fitness = {best_fitness:.5f}')

    # Selection
    selected_population = selection(population, fitness_values)

    # Crossover
    next_generation = []
    for i in range(0, population_size, 2):
        parent1 = selected_population[i]
        if i + 1 < population_size:
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])
        else:
            next_generation.append(parent1)

    # Mutation
    population = [mutate(chromosome) for chromosome in next_generation]

# Find the best individual from the last generation
fitness_values = evaluate_fitness(population)
best_fitness = max(fitness_values)
best_individual = population[fitness_values.index(best_fitness)]
best_x = decode(best_individual)

print('\\nBest solution found:')
print(f'x = {best_x:.5f}')
print(f'Fitness = {best_fitness:.5f}')

# Plot the evolution of the best fitness
plt.plot(best_fitness_history)
plt.title('Best Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()

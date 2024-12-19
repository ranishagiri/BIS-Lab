import numpy as np
import matplotlib.pyplot as plt
import random

# Number of cities
n_cities = 50

# Randomly generate city coordinates
np.random.seed(42)
cities = np.random.rand(n_cities, 2) * 100

# Distance matrix between cities
def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

distance_matrix = np.array([[distance(c1, c2) for c2 in cities] for c1 in cities])

# Ant Colony Optimization Parameters
n_ants = 20
n_iterations = 100
alpha = 1.0    # Pheromone importance
beta = 1.0     # Distance importance
evaporation_rate = 0.9
Q = 100        # Constant related to pheromone deposition
initial_pheromone = 1.0

# Initialize pheromone matrix
pheromone_matrix = np.ones((n_cities, n_cities)) * initial_pheromone

# Function to choose the next city for an ant based on probability
def select_next_city(current_city, visited, pheromone_matrix, distance_matrix, alpha, beta):
    probabilities = []
    for i in range(n_cities):
        if i not in visited:
            pheromone = pheromone_matrix[current_city][i] ** alpha
            distance = (1 / distance_matrix[current_city][i]) ** beta
            probabilities.append(pheromone * distance)
        else:
            probabilities.append(0)

    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return np.random.choice(range(n_cities), p=probabilities)

# ACO main loop
best_route = None
best_length = float('inf')

for iteration in range(n_iterations):
    routes = []
    lengths = []

    # Each ant builds a tour
    for ant in range(n_ants):
        current_city = random.randint(0, n_cities - 1)
        route = [current_city]
        visited = set(route)

        # Build a complete tour
        for _ in range(n_cities - 1):
            next_city = select_next_city(current_city, visited, pheromone_matrix, distance_matrix, alpha, beta)
            route.append(next_city)
            visited.add(next_city)
            current_city = next_city

        # Complete the tour by returning to the starting city
        route.append(route[0])
        routes.append(route)

        # Calculate the length of the tour
        length = sum([distance_matrix[route[i]][route[i+1]] for i in range(n_cities)])
        lengths.append(length)

        # Update the best route
        if length < best_length:
            best_length = length
            best_route = route

    # Update pheromone levels
    pheromone_matrix *= (1 - evaporation_rate)  # Evaporate pheromone

    # Add pheromone to the routes based on their quality
    for route, length in zip(routes, lengths):
        for i in range(n_cities):
            pheromone_matrix[route[i]][route[i+1]] += Q / length

    # Print progress
    print(f"Iteration {iteration+1}/{n_iterations}, Best Length: {best_length}")

# Plot the best route
def plot_route(cities, route):
    plt.figure(figsize=(10, 5))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='Cities')
    for i, city in enumerate(cities):
        plt.annotate(f'{i}', (city[0], city[1]))

    for i in range(n_cities):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'r-')

    plt.title('Best Route Found by ACO')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

# Display the best route
print("R Abhinav\n1BM22CS211")
print("Best route:", best_route)
plot_route(cities, best_route)

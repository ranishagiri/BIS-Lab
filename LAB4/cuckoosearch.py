import numpy as np

def objective_function(x):
    # Example objective function: Sphere function
    return np.sum(x**2)

def levy_flight(Lambda):
    # Generate a step size following a Lévy distribution
    sigma_u = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
               (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn() * sigma_u
    v = np.random.randn()
    step = u / abs(v) ** (1 / Lambda)
    return step

def cuckoo_search(num_nests=25, max_iter=100, pa=0.25, lb=-5, ub=5, dim=2):
    # Initialize nests
    nests = np.random.uniform(lb, ub, (num_nests, dim))
    fitness = np.array([objective_function(n) for n in nests])
    
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    for iteration in range(max_iter):
        new_nests = np.copy(nests)
        
        # Lévy flights and update nests
        for i in range(num_nests):
            step_size = levy_flight(1.5)
            step = step_size * (nests[i] - best_nest)
            new_nests[i] = nests[i] + step * np.random.randn(dim)
            new_nests[i] = np.clip(new_nests[i], lb, ub)
        
        # Evaluate fitness for new nests
        new_fitness = np.array([objective_function(n) for n in new_nests])
        
        # Replace some of the nests based on the fitness
        for i in range(num_nests):
            if new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        
        # Sort and update the best nest
        min_fitness_index = np.argmin(fitness)
        if fitness[min_fitness_index] < best_fitness:
            best_nest = nests[min_fitness_index]
            best_fitness = fitness[min_fitness_index]
        
        # Abandon some nests with probability pa and create new nests
        for i in range(num_nests):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(lb, ub, dim)
                fitness[i] = objective_function(nests[i])
        
        # Display the best fitness at each iteration
        print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {best_fitness}")
    
    return best_nest, best_fitness

# Usage
print("R Abhinav\n1BM22CS211")
best_solution, best_value = cuckoo_search()
print(f"best solution: {best_solution}, best value: {best_value}")

import numpy as np

# Define the objective function to minimize
def objective_function(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)  # Rastrigin Function (benchmark for optimization algorithms)

class Particle:
    def __init__(self, dimensions):
        self.position = np.random.uniform(-100, 100, dimensions)  # Initialize positions
        self.velocity = np.random.uniform(-10, 10, dimensions)    # Initialize velocities
        self.best_position = np.copy(self.position)             # Best position of the particle
        self.best_score = objective_function(self.position)     # Best score of the particle

    def update_velocity(self, global_best_position, w=0.729, c1=1.49445, c2=1.49445):
        inertia = w * self.velocity
        cognitive_component = c1 * np.random.random() * (self.best_position - self.position)
        social_component = c2 * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity

        # Keep particles within bounds (optional, depending on problem)
        self.position = np.clip(self.position, -10, 10)

class ParticleSwarmOptimizer:
    def __init__(self, dimensions, num_particles=30, iterations=100):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.iterations = iterations
        self.particles = [Particle(dimensions) for _ in range(num_particles)]
        self.global_best_position = np.copy(self.particles[0].position)
        self.global_best_score = objective_function(self.global_best_position)

    def optimize(self):
        for iteration in range(self.iterations):
            for particle in self.particles:
                score = objective_function(particle.position)

                # Update the personal best of the particle
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)

                # Update the global best position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(particle.position)

            # Update velocity and position of each particle
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            print(f"Iteration {iteration + 1}/{self.iterations}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score

# Example usage:
if __name__ == "__main__":
    print("R Abhinav\n1BM22CS211")
    dimensions = 5  # Number of dimensions of the search space
    pso = ParticleSwarmOptimizer(dimensions, num_particles=50, iterations=100)
    best_position, best_score = pso.optimize()
    
    print("Best position:", best_position)
    print("Best score:", best_score)

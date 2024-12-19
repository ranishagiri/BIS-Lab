import numpy as np

# Define the objective function (e.g., Sphere function)
def objective_function(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

# Grey Wolf Optimizer
class GreyWolfOptimizer:
    def __init__(self, obj_func, dim, n_wolves=20, max_iter=100, lb=-10, ub=10):
        self.obj_func = obj_func
        self.dim = dim  # Dimension of the problem
        self.n_wolves = n_wolves  # Number of wolves (population size)
        self.max_iter = max_iter  # Maximum number of iterations
        self.lb = lb  # Lower bound
        self.ub = ub  # Upper bound

        # Initialize the wolves randomly
        self.positions = np.random.uniform(lb, ub, (n_wolves, dim))
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

    def optimize(self):
        for t in range(self.max_iter):
            # Update the positions of alpha, beta, and delta
            for i in range(self.n_wolves):
                fitness = self.obj_func(self.positions[i])

                # Update alpha, beta, delta based on fitness
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            # Update each wolf's position
            a = 2 - t * (2 / self.max_iter)  # Linearly decreases from 2 to 0
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # Update wolf position
                    self.positions[i, j] = (X1 + X2 + X3) / 3

                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.alpha_score, self.alpha_pos

# Example usage
dim = 5  # Number of dimensions
optimizer = GreyWolfOptimizer(objective_function, dim=dim, n_wolves=20, max_iter=100, lb=-10, ub=10)
best_score, best_position = optimizer.optimize()
print(f"best score: {best_score}; best position: {best_position}")
print("R Abhinav\n1BM22CS211")

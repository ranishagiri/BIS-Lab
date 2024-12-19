import random 

# Number of individuals in each generation 
POPULATION_SIZE = 100

# Valid genes 
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated 
TARGET = "I love GeeksforGeeks"

class Individual:
    ''' 
    Class representing individual in population 
    '''
    def __init__(self, chromosome): 
        self.chromosome = chromosome 
        self.fitness = self.cal_fitness() 

    @staticmethod
    def mutated_genes(): 
        ''' 
        Create random genes for mutation 
        '''
        return random.choice(GENES)

    @staticmethod
    def create_gnome(): 
        ''' 
        Create chromosome or string of genes 
        '''
        gnome_len = len(TARGET) 
        return [Individual.mutated_genes() for _ in range(gnome_len)] 

    def mate(self, par2): 
        ''' 
        Perform mating and produce new offspring 
        '''
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):     
            prob = random.random() 
            if prob < 0.45: 
                child_chromosome.append(gp1) 
            elif prob < 0.90: 
                child_chromosome.append(gp2) 
            else: 
                child_chromosome.append(self.mutated_genes()) 

        return Individual(child_chromosome) 

    def cal_fitness(self): 
        ''' 
        Calculate fitness score, it is the number of 
        characters in string which differ from target 
        string. 
        '''
        fitness = sum(gs != gt for gs, gt in zip(self.chromosome, TARGET))
        return fitness 

# Driver code 
def main(): 
    generation = 1
    found = False
    population = [] 

    # Create initial population 
    for _ in range(POPULATION_SIZE): 
        gnome = Individual.create_gnome() 
        population.append(Individual(gnome)) 

    while not found: 
        # Sort the population in increasing order of fitness score 
        population = sorted(population, key=lambda x: x.fitness) 

        # Check if the target is reached 
        if population[0].fitness <= 0: 
            found = True
            break

        new_generation = [] 

        # Elitism: 10% of fittest population goes to the next generation 
        s = int((10 * POPULATION_SIZE) / 100) 
        new_generation.extend(population[:s]) 

        # Generate offspring from the next 90% of fittest population 
        s = int((90 * POPULATION_SIZE) / 100) 
        for _ in range(s): 
            parent1 = random.choice(population[:50]) 
            parent2 = random.choice(population[:50]) 
            child = parent1.mate(parent2) 
            new_generation.append(child) 

        # Ensure the new generation size is POPULATION_SIZE
        population = new_generation[:POPULATION_SIZE] 

        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}") 

        generation += 1

    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}") 
print("R Abhinav\n1BM22CS211")
if __name__ == '__main__': 
    main()

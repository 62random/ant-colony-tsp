import numpy as np 

# Nearest neighbour algorithm so solve an optimization problem
# with a greedy approach
class NearestNeighbour():
    def __init__(self, env, substitution=False):
        self.env = env 
        self.substitution = substitution

    def run(self, start_point=0):
        distances = self.env.d.copy()
        total_distance = 0

        solution = np.zeros(self.env.max_solution_size(), dtype=int)
        solution[0] = start_point
        solution[-1] = start_point
        for i in range(distances.shape[0]):
            distances[i][i] = np.inf
        
        for i in range(self.env.solution_steps()):
            next_node = np.argmin(distances[solution[i]])
            solution[i+1] = next_node
            total_distance += self.env.d[solution[i]][next_node]
            distances[:,solution[i]] = np.inf
        
        total_distance += self.env.d[solution[-1]][solution[-2]]
            
        return solution, total_distance
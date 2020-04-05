import numpy as np
import random
import sys

# Abstract class for applying the Ant Colony Optimization strategy
# to a given problem. Some assumptions are made, so the representation
# of the problem must meaningfully implement (at least) the functions
# called from the self.env variable (env for environment)
class AntColonyOptimizer():
    # Instantiation of the optimizer
    def __init__(self, env, n_ants=100, evap_rate=0.5, alpha=1, beta=2, substitution=False):
        self.n_ants = n_ants
        self.evap_rate = evap_rate 
        self.alpha = alpha 
        self.beta = beta
        self.env = env
        # Substitution -> if we can repeat steps in our solutions (we can't in TSP, for example)
        self.substitution = substitution


    # Finding a solution in a given number of iterations or if the used evaluation metric
    # does not improve in a given number of consecutive steps
    def run(self, max_iter=100, start_point=0, random_start=False, convergence_steps=10):
        
        # Initialize pheromone distribution and solutions
        pheromone = .1 * np.ones(self.env.step_space())
        solutions = np.ones((self.n_ants, self.env.max_solution_size()), dtype=int)
        metric_records = np.empty(max_iter)
        metric_records[:] = np.nan


        # Current best solution metric value (distance in the case of tsp, for example)
        alltime_best_metric = self.env.initial_metric_value()
        convergence_counter = 0

        # Loop until converges or until reaches max iterations
        for iteration in range(max_iter):

            # Loops through every ant
            for i in range(self.n_ants):
                # Set starting point for solution
                solutions[i, -1] = solutions[i, 0] = self.env.first_step(start_point, random_start)
                attractiveness = self.env.attractiveness()

                # Build the solution step by step
                for j in range(self.env.solution_steps()):
                    current_location = solutions[i, j]
                    # Make sure we don't repeat steps in the solution
                    if not self.substitution:
                        attractiveness[:, current_location] = 0

                    # Formula to chose next step in ACO
                    # Each variable name represents it's role in the formula
                    pheromone_factor = np.power(pheromone[current_location, :], self.alpha)
                    attractiveness_factor = np.power(attractiveness[current_location, :], self.beta)
                    product = np.multiply(pheromone_factor, attractiveness_factor)
                    total = np.sum(product)
                    probabilities = np.divide(product, total)

                    # To choose the next step
                    cum_prob = np.cumsum(probabilities)
                    r = random.random()
                    solutions[i, j + 1] = np.nonzero(cum_prob > r)[0][0]

            # Calculating the solution evaluations
            solution_evaluations = np.zeros(self.n_ants)
            for i in range(self.n_ants):
                solution_evaluations[i] = self.env.evaluate(solutions[i])
            
            # Pick best solution in this iteration
            best_solution_index = np.argmin(solution_evaluations)
            best_metric_iteration = solution_evaluations[best_solution_index]
            best_solution_iteration = solutions[best_solution_index]
            # If this is a new record, save the solution!
            if best_metric_iteration < alltime_best_metric:
                alltime_best_metric = best_metric_iteration
                alltime_best_solution = best_solution_iteration
                convergence_counter = 0
            else: 
                convergence_counter += 1

            # To later plot the convergence of the algorithm
            metric_records[iteration] = alltime_best_metric
            
            if convergence_counter > convergence_steps:
                return alltime_best_solution, metric_records[np.logical_not(np.isnan(metric_records))]

            # Update pheromone spreading
            pheromone = (1 - self.evap_rate) * pheromone
            for i in range(self.n_ants):
                for j in range(self.env.solution_steps()):
                    dt = self.env.pheromone_variation(solution_evaluations[i])
                    pheromone[solutions[i, j], solutions[i, j+1]] += dt

        return alltime_best_solution, metric_records[np.logical_not(np.isnan(metric_records))]
            

            







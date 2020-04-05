import numpy as np 
from PIL import Image, ImageDraw, ImageColor
from IPython.display import display
import random 
from math import sqrt
import sys


# Class that describes the Traveling Salesman Problem
# Utility class including some features to support the
# Solution of this problem with a given strategy, for example
# Ant Colony Optimization
class TSP:
    def __init__(self, n_nodes, height=20, width=20):
        self.n_nodes = n_nodes
        self.height = height
        self.width = width

        nodes = []
        for _ in range(n_nodes):
            nodes.append((random.random()*width, random.random()*height))

        self.nodes = np.array(nodes, dtype='float16')

        d = []
        for i in self.nodes:
            l = []
            for j in self.nodes:
                l.append(sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2))
            d.append(l)
        self.d = np.array(d, dtype='float64')

    # The shape of the space of possible combinations for steps that
    # constitute valid solutions
    def step_space(self):
        return self.n_nodes, self.n_nodes

    # Initial (worst possible) value for metric that we want to
    # optimize to indicate that we found a good solution
    def initial_metric_value(self, metric='distance'):
        if metric == 'distance':
            return sys.float_info.max

    # Returns the first step in a given solution
    def first_step(self, start_point, random_start=False):
        if random_start:
            if random.random() > 0.5:
                return start_point
            else:
                return random.randint(0, self.n_nodes - 1)
        else:
            return start_point

    # Attractiveness of the possible steps
    # in the context of approaches like Ant Colony Optimization
    def attractiveness(self):
        arr = np.divide(1, self.d)
        arr[arr == np.inf] = 0
        return arr

    # Returns the number of steps required to construct a solution
    def solution_steps(self):
        return self.n_nodes - 1

    # Maximum size for a valid solution to the problem
    def max_solution_size(self):
        return self.n_nodes + 1

    # Evaluates a solution to the problem
    # Returns the distance of the path
    def evaluate(self, solution):
        distance = 0
        for j in range(len(solution) - 1):
            distance += self.d[solution[j]][solution[j + 1]]

        return distance

    # Returns the pheromonal variation according to an evaluation
    # Only makes sense when using Ant Colony Optimization
    def pheromone_variation(self, evaluation):
        return 1/evaluation

    # Renders the environment and, if provided, solution paths
    def render(self, dim=600, paths=None):
        window_width = dim
        window_height = int((self.height/float(self.width))*window_width)
        image = Image.new(mode='RGBA', size=(window_width, window_height), color='white')

        draw = ImageDraw.Draw(image, 'RGBA')

        psize = (window_width/(self.n_nodes*5))
        for n in self.nodes:
            draw.ellipse([n[0]*window_width/self.width - psize, n[1]*window_height/self.height - psize, 
                            n[0]*window_width/self.width + psize, n[1]*window_height/self.height + psize], fill='black')
        
        if paths:
            colors = ['red', 'blue']
            for j in range(len(paths)):
                path = paths[j]
                for i in range(len(path) - 2):
                    p0 = self.nodes[path[i]] 
                    p1 = self.nodes[path[i+1]] 
                    draw.line([p0[0]*window_width/self.width - 3 * j, 
                               p0[1]*window_height/self.height - 3 * j,
                               p1[0]*window_width/self.width - 3 * j, 
                               p1[1]*window_height/self.height - 3 * j],
                               fill=colors[j % len(colors)], width=2)

        del draw
        return display(image)
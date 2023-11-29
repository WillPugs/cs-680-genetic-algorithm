"""
File defining the class GeneticAlgorithmResult, which holds the results of running a GA to solve a ILP (specifically the knapsack problem).
"""
import numpy as np

class GeneticAlgorithmResult:
    """ The result of running a genetic algorithm to optimize an ILP.
    Contains the following attributes:
        - opt_val: the estimated optimal value
        - opt_sol: the estimates optimal solution
        - values: the smallest value of each generation
        - selection: the selection algorithm used
        - crossover: the number of crossover points
        - mutation_rate: the mutation rate
        - iterations: the number of before stopping
        - running_time: The time it took to run the algorithm
    """
    def __init__(self,
                 opt_val=None,
                 opt_sol=None,
                 values=None,
                 selection=None,
                 crossover=None,
                 mutation_rate=None,
                 iterations=None,
                 running_time=None):
        self.opt_val = opt_val
        self.opt_sol = opt_sol
        self.values = values
        self.selection = selection
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.running_time = running_time
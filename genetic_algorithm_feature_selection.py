"""
File defining the class GeneticAlgorithm, which creates and runs GAs to solve ILPs (specifically the knapsack problem).

Running this file with no additional arguments will run the tests defined at the bottom.
"""
import numpy as np
import time
from offspring import *
from genetic_algorithm_result import *
from sklearn.ensemble import BaggingClassifier


class GeneticAlgorithm:
    algorithm_names = {0: "Random With Replacement",
                            1: "Random Without Replacement",
                            2: "Rank-Based",
                            3: "Tournament Selection",
                            4: "Truncation"}
    
    def __init__(self, X, Y, model, P, n_estimators=None, max_samples=None, cv=None):
        """ (self, np.array, np.array, sklearn_model) -> (GeneticAlgorithm)
        X: features
        Y: labels
        """
        self.X = X
        self.Y = Y
        self.model = model

        if (n_estimators is not None) and (max_samples is not None) and (cv is not None):
            raise ValueError("Can only specify a genetic algorithm using either bagging or cross-validation, not both.")
        self.P = P
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.cv = cv

        self.n = X.shape[0] #number of data points
        self.d = X.shape[1] #number of features

        self.population = None #array of the individuals
        self.fitnesses = None #fitness values of the individuals
        self.optimal_solution = None #the current optimal solution
        self.optimal_value = 0 #the current optimal value
        self.values = np.array([]) #the optimal value at each iteration of the algorithm

    #getters and setters
    def get_X(self):
        return self.X
    def set_X(self, X):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
    def get_Y(self):
        return self.Y
    def set_P(self, Y):
        self.Y = Y

    #these values should not change
    def get_n(self):
        return self.n
    def get_d(self):
        return self.d
    def get_P(self):
        return self.P

    
    
    def get_fitnesses(self):
        return self.fitnesses
    def set_fitnesses(self, fitnesses):
        self.fitnesses = fitnesses


    def get_population(self):
        return self.population
    def set_population(self, population):
        self.population = population
        
        #update other parameters
        fitnesses = []
        biggest_value = self.fitness(population[0]) #initial guess for best fitness value in population
        for individual in population:
            #update optimal values
            fit_val = self.fitness(individual)
            if fit_val > self.get_optimal_value(): #better than current best
                self.set_optimal_solution(individual)
                self.set_optimal_value(fit_val)
            if fit_val > biggest_value:
                biggest_value = fit_val
            fitnesses.append(fit_val)
        self.set_fitnesses(np.array(fitnesses))
        self.append_values(biggest_value)
            

    def get_optimal_solution(self):
        return self.optimal_solution
    def set_optimal_solution(self, sol):
        self.optimal_solution = sol

    def get_optimal_value(self):
        return self.optimal_value
    def set_optimal_value(self, val):
        self.optimal_value = val
    
    def get_values(self):
        return self.values
    def append_values(self, new_val):
        self.values = np.append(self.values, new_val)

    #utility methods
    def fitness(self, solution):
        """ (self, np.array) -> (num)
        Performs cv-fold cross-validation.
        """
        if self.cv is None: #do not use cross-validation
            return self.fitness_bagging(solution)
        if np.sum(solution)==0: #no features
            return 0 #terrible score
        num_points = self.get_n()//self.get_cv()
        data_point_indices = np.arange(self.get_n())
        np.random.shuffle(data_point_indices)

        scores = 0
        for i in range(self.get_cv()):
            #train
            fold_indices = data_point_indices[i*num_points:num_points*(i+1)]
            train_indices = np.setdiff1d(data_point_indices, fold_indices)
            self.model.fit(self.get_X()[train_indices,:][:,solution.astype(bool)], self.get_Y()[train_indices])
            
            #test
            scores += self.model.score(self.get_X()[fold_indices,:][:,solution.astype(bool)], self.get_Y()[fold_indices])

        return scores/self.get_cv()

    def fitness_bagging(self, solution):
        """ (self, np.array) -> (num)
        Performs bagging classification on x_train
        """
        if np.sum(solution)==0: #no features
            return 0 #terrible score
        solution_model = BaggingClassifier(estimator=self.model, n_estimators=self.n_estimators, max_samples=self.max_samples)
        solution_model.fit(self.get_X()[:,solution.astype(bool)], self.get_Y())
        return solution_model.score(self.get_X()[:,solution.astype(bool)], self.get_Y())
        


    def initialize(self):
        """ (self) -> ()
        Starts the algorithm with an initial guess for each member of the population.
        """
        pop = np.random.randint(0, 2, (self.get_P(), self.get_d()))
        #pop[0,:] = np.ones(self.get_d())

        self.set_population(pop)
    

    def sort_population(self):
        """ (self) -> ()
        Sorts the population according to the individuals' fitness values.
        Smallest to highest
        """
        p = self.get_fitnesses().argsort()
        self.fitnesses = self.get_fitnesses()[p]
        self.population = self.get_population()[p,:]
        

    #selection algorithms
    def select(self, selection, num_children, tournament_k=None):
        """ (self, int, int, int) -> (np.array)
        Selects the members of the population used for reproduction.

        selection: Selection algorithm to use
            - 0: random with replacement
            - 1: random without replacement
            - 2: rank-based
            - 3: tournament selection
            - 4: truncation
        num_children: The number of children needed
            - if num_children is odd, the number of parents needed is one more than the number of children
            - if num_children is even, the number of parents needed is the number of children
        """
        if selection==0:
            return self.random_select(num_children, replacement=True)
        elif selection==1:
            return self.random_select(num_children, replacement=False)
        elif selection==2:
            return self.rank_select(num_children)
        elif selection==3:
            return self.tournament_select(num_children, tournament_k)
        elif selection==4:
            return self.truncation_select(num_children)
        else:
            raise ValueError(f"{selection} is not a valid input for a selection algorithm.")
    

    def random_select(self, num_children, replacement):
        """ (self, int, boolean) -> (np.array)
        Randomly selects from the population for the individuals to be used for reproduction.

        - replacement: Random selection with replacement
        """
        if num_children%2==1:
            indices = np.random.choice(np.arange(self.get_P()), size=num_children+1, replace=replacement)
            return self.get_population()[indices,:]
        else:
            indices = np.random.choice(np.arange(self.get_P()), size=num_children, replace=replacement)
            return self.get_population()[indices,:]
        

    def rank_select(self, num_children):
        """ (self, int) -> (np.array)
        Selects the parents based on the fitness of the individuals normalized by the total fitness.
        Probability is proportional to fitness rank (the most fit has probability n/(1+...+n))
        """
        ranks = np.array([i+1 for i in range(self.get_P())])
        ranks = ranks/np.sum(ranks)
        if num_children%2==1:
            indices = np.random.choice(np.arange(self.get_P()), size=num_children+1, replace=False, p=ranks)
            return self.get_population()[indices,:]
        else:
            indices = np.random.choice(np.arange(self.get_P()), size=num_children, replace=False, p=ranks)
            return self.get_population()[indices,:]
    

    def tournament_select(self, num_children, tournament_k):
        """ (self, int, int) -> (np.array)
        Selects tournament_k members from the population, the most fit is selected to be a parent.
        """
        #the number of parents we want
        if num_children%2==1: #odd
            num_parents = num_children+1
        else:
            num_parents = num_children
        
        #first candidate, so that we can use vstack in the loop
        indices = np.random.choice(np.arange(self.get_P()), size=tournament_k, replace=False)
        candidates = self.get_population()[indices,:]
        candidate_fitness = self.get_fitnesses()[indices]
        parents = np.array([candidates[np.argmax(candidate_fitness)]])

        while parents.shape[0]<num_parents: #number of rows (parents) is less than the desired number
            indices = np.random.choice(np.arange(self.get_P()), size=tournament_k, replace=False)
            candidates = self.get_population()[indices,:]
            candidate_fitness = self.get_fitnesses()[indices]
            parents = np.vstack((parents, candidates[np.argmax(candidate_fitness)]))
        
        return parents
    

    def truncation_select(self, num_children):
        """ (self, int) -> (np.array)
        Picks the parents in order of most fit.
        """
        #the number of parents we want
        if num_children%2==1: #odd
            num_parents = num_children+1
        else:
            num_parents = num_children
        return self.get_population()[:self.get_P()-num_parents,:]
    
    #create children
    def create_children_population(self, parents, num_children, crossover):
        """ (self, 2D np.array, int, int) -> (2D np.array)
        The parents will reproduce to create num_children children.
        """
        np.random.shuffle(parents) #will shuffle rows
        #first child so we can use vstack in loop
        child1, child2 = ChildSolution(parents[0], parents[1], crossover).create_children()
        children = np.array([child1, child2])

        i = 2
        while children.shape[0] < num_children:
            child1, child2 = ChildSolution(parents[i], parents[i+1], crossover).create_children()
            children = np.vstack((children, child1, child2))
            i += 2

        #drop last child if we're looking for an odd number of children
        return children[0:num_children,:]


    #replace individiuals in pop with children
    def replacement(self, children):
        """ (self, 2D np.array) -> ()
        Replaces the least fit adults with the children.
        """
        current_pop = self.get_population()
        current_fitness = self.get_fitnesses()

        #the population will be sorted every time replacement is called
        i = 0
        for child in children:
            current_pop[i,:] = child
            i += 1        
        self.set_population(current_pop)

    #mutation
    def mutate(self, children, mutation_rate):
        """ (self, np.array, num) -> (np.array)
        Mutates the children with probability mutation_rate.
        """
        return (children + np.random.random(children.shape)<=mutation_rate) % 2



    def replace_duplicates(self, unique_individuals, replace_occurrence):
        """ (self, np.array) -> ()
        Replaces the duplicate individuals in the population.
        """
        counts = np.array([])
        for indv in unique_individuals:
            count = 0
            for row in self.get_population():
                if np.all(row==indv):
                    count += 1
            counts = np.append(counts, count)
        counts = counts.astype(np.int64)
        #counts = np.array([np.sum(self.get_population()==indv) for indv in unique_individuals])


    #overall algorithm
    def run(self, 
            iterations=1000, 
            mutation_rate=0.01, 
            crossover=1, 
            selection=0, 
            num_children=1, 
            tournament_k=None,
            replace_occurrence=100,
            replace_amount=1,):
        """ (self, int, num, int int, int, int) -> (GeneticAlgorithmResult)
        Runs the genetic algorithm on the ILP given by the class' attributes.

        mutation_rate: mutation rate (num)
        crossover: the number of crossover points (int)
        selection: which selection algorithm to use (int)
            - 0: random with replacement
            - 1: random without replacement
            - 2: rank-based
            - 3: tournament selection
            - 4: truncation
        num_children: number of children to create each generation
        tournament_k: the number of parents per round of selection in tournament selection
        replace_occurrence:
        replace_amount: 

        Returns GeneticAlgorithmResult with the following attributes:
            - opt_val: the estimated optimal value
            - opt_sol: the estimates optimal solution
            - values: the smallest value of each generation
            - selection: the selection algorithm used
            - crossover: the number of crossover points
            - mutation_rate: the mutation rate
            - iterations: the number of before stopping
            - running_time: The time it took to run the algorithm
        """
        #input validation
        if num_children > self.get_P():
            raise ValueError("The number of children replacing the current generation cannot be greater than the generation size.")
        if selection not in [0, 1, 2, 3, 4]:
            raise ValueError(f"{selection} is not a valid input for a selection algorithm.")
        if tournament_k is None and selection==3:
            raise ValueError("When tournament selection is chosen as the selection algorithm, the tournament_k parameter must be specified.")
        
        #record running time
        t_start = time.perf_counter()

        #initialize
        self.initialize()

        i = 0
        while i<iterations:
            if i%replace_occurrence==0 and False: #every hundred iterations check for similarity
                unique_individuals = np.unique(self.get_population(), axis=0)
                if unique_individuals.shape[0] <= self.get_P()/2: #more than half are duplicates
                    self.replace_duplicates(unique_individuals, replace_amount)
                else:
                    self.sort_population()
                    #selection
                    parents = self.select(selection=selection, num_children=num_children, tournament_k=tournament_k)

                    #create children
                    children = self.create_children_population(parents, num_children, crossover)
                    
                    #mutation
                    children = self.mutate(children, mutation_rate)
                    
                    #update population, fitnesses, values, and optimality
                    self.replacement(children)

                    #stopping conditions     

                #next iteration
                i += 1   
            else:
                if i%(iterations//10)==0:
                    pass
                    #print(f"{100*i/iterations}% iterations ran")
                self.sort_population()
                #selection
                parents = self.select(selection=selection, num_children=num_children, tournament_k=tournament_k)

                #create children
                children = self.create_children_population(parents, num_children, crossover)
                
                #mutation
                children = self.mutate(children, mutation_rate)
                
                #update population, fitnesses, values, and optimality
                self.replacement(children)      

                #stopping conditions

                #next iteration
                i += 1

        t_stop = time.perf_counter()
        
        results = GeneticAlgorithmResult(self.get_optimal_value(),
                                         self.get_optimal_solution(),
                                         self.get_values(),
                                         selection,
                                         crossover,
                                         mutation_rate,
                                         i,
                                         t_stop-t_start)
        return results
    
    @staticmethod
    def values_to_optimal_values(values):
        """ (np.array) -> (np.array)
        Converts from an array of the largest value at each iteration (this is the values attribute of a GeneticAlgorithm instance)
        to an array of largest value out of any generation so far.
        """
        opt_vals = np.array([values[0]])
        for i in range(1, values.size):
            opt_vals = np.append(opt_vals, max(opt_vals[-1], values[i]))
        return opt_vals


if __name__ == '__main__':
    test_alg = True
    if test_alg:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(n_samples=200, n_features=10, n_redundant=5, n_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        test_GA = GeneticAlgorithm(X_train, y_train, DecisionTreeClassifier(), 10, n_estimators=5, max_samples=50)
        result = test_GA.run(100, 0.01, 2, 0, 3, None, 100, 1)
        print(f"Running time: {result.running_time} s")
        print(f"Optimal solution: {result.opt_sol}")
        print(f"Optimal score: {result.opt_val}")
        #print("Largest fitness value at each generation:")
        #print(result.values)
        
        model = DecisionTreeClassifier()

        model.fit(X_train[:,result.opt_sol.astype(bool)], y_train)
        print(f"Model trained on selected features: {model.score(X_test[:,result.opt_sol.astype(bool)], y_test)}")

        model.fit(X_train, y_train)
        print(f"Model trained on all features: {model.score(X_test, y_test)}")
        #print(GeneticAlgorithm.algorithm_names[0])
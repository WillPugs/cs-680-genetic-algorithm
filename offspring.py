"""
File defining the class ChildSolution, which is used to implement the crossover algorithm when creating a child chromosome from two parents.

Running this file with no additional arguments will run the tests defined at the bottom.
"""
import numpy as np


class ChildSolution:

    def __init__(self, mother, father, crossover):
        """ (self, np.array, np.array, int) -> (ChildSolution)
        
        mother: one of the the parents (np.array)
        father: the other parent (np.array)
        crossover: the number of crossover points (int)
        """
        if mother.size != father.size:
            raise AttributeError("The two parent chromosomes must have the same size.")
        self.mother = mother
        self.father = father

        if crossover > mother.size:
            raise AttributeError("The number of crossover points cannot be greater than the size of the parent arrays.")
        self.crossover = crossover

    #crossover algorithm
    def create_children(self):
        """ (self) -> (np.array, np.array)
        Creates two children by crossing over the mother and father chromosomes.
        """
        #initialize with an array of -1s
        child1 = -np.ones_like(self.mother)
        child2 = -np.ones_like(self.father)

        if self.crossover == self.mother.size:
            #special case, no need to sort random values
            indices = np.random.randint(0, 2, self.crossover)
            child1[indices==0] = self.mother[indices==0]
            child1[indices==1] = self.father[indices==1]

            child2[indices==1] = self.mother[indices==1]
            child2[indices==0] = self.father[indices==0]
        else:
            #pick crossover points, sort in increasing order
            indices = np.sort(np.random.choice(self.mother.size, self.crossover, replace=False))
            indices = np.append(indices, self.mother.size)

            #iterate through points and distribute
            child1[0:indices[0]] = self.mother[0:indices[0]]
            child2[0:indices[0]] = self.father[0:indices[0]]
            for i in range(1, indices.size):
                if i%2 == 1:
                    child1[indices[i-1]:indices[i]] = self.father[indices[i-1]:indices[i]]
                    child2[indices[i-1]:indices[i]] = self.mother[indices[i-1]:indices[i]]
                else:
                    child1[indices[i-1]:indices[i]] = self.mother[indices[i-1]:indices[i]]
                    child2[indices[i-1]:indices[i]] = self.father[indices[i-1]:indices[i]]

        return child1, child2
    

if __name__ == "__main__":
    testing = True
    if testing:
        mother = np.array([1, 2, 3, 4, 5, 6])
        father = 10*mother
        for i in range(100):
            child_sol = ChildSolution(mother, father, mother.size)
            print(child_sol.create_children())
import random

from deap import base
from deap import creator
from deap import tools
import os
from PIL import Image

import numpy as np
from fitness import fitness_for_a_pic
import matplotlib.pyplot as plt
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
# Attribute generator 
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 2500)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):

    pic = np.array(individual)
    pic = np.reshape(pic , (pic.shape[0], 50,50,1))
    print(pic.shape)
    
    loss = fitness_for_a_pic(pic , 3)
    return loss


def plotter(pic):
    pic = np.array(pic)
    pic = np.reshape(pic , (50,50,1))
    plt.imshow(pic)
    plt.show()

def load_some_shit(number = 3 , count = 10):
    path = './Train/Original/'+ str(number) +'/'
    X = []
    total = 0
    for filename in os.listdir(path):
        if(filename!='.DS_Store'):
            im = Image.open(path+filename)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x<128 else 1)
            pix = np.array(im)
            pix = np.reshape(pix , (2500,))
            X.append(pix)
            total +=1
            if(total >= count):
                break
    X = np.array(X)
    return X
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=300)
    count = 10
    init_pop = load_some_shit(count= count)
    for i in range(count):
        for j in range(init_pop.shape[1]):
            pop[i][j] = init_pop[i][j]
    # print(init_pop.shape)
    # exit(1)
    fitnesses = evalOneMax(pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit, 
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = map(toolbox.evaluate, invalid_ind)
        fitnesses = evalOneMax(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit,
       
        pop[:] = offspring
       
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        if(g %10 == 1):
            mn = max(fits)
            for ind, fit in zip(pop, fits):
                if fit == mn :
                    plotter(ind)
                    break

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

if __name__ == '__main__': 
    main()
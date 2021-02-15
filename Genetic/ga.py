import random

from deap import base
from deap import creator
from deap import tools
import os
from PIL import Image
import numpy as np
from fitness import fitness_for_a_pic
import matplotlib.pyplot as plt

import cv2



def counter(img):
    contours, hierarchy= cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    number_of_objects_in_image= len(contours)
    return number_of_objects_in_image


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
    # print(pic.shape)
    # exit(1)
    loss = fitness_for_a_pic(pic , 3)
    for i in range(pic.shape[0]):
        img = pic[i,:]
        if counter(1- np.squeeze(img)) > 1:
            loss[i] = -500
    return loss


def plotter(pic):
    pic = np.array(pic)
    pic = np.reshape(pic , (50,50))
    plt.imshow(pic)
    plt.show()
    plt.close()

def load_some_pics(number = 3 , count = 10):
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

new_Pop = []

def main():
    pop = toolbox.population(n=300)
    count = 300
    init_pop = load_some_pics(count= count)
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
    CXPB, MUTPB = 0.3, 0.1
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0
    # Begin the evolution
    wt = []
    rt = []
    cntt = 1
    while max(fits) < 100 and g < 301:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals


        if (g == 301):
            new_pop = np.array(new_Pop)
            for i in range(count):
                for j in range(new_pop.shape[1]):
                    pop[i][j] = new_pop[i][j]
            
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
        
        mn = max(fits)
        for ind, fit in zip(pop, fits):
            if fit == mn :
                # plotter(ind)
                new_Pop.append(ind)
                # break

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        rm = []
        for q in fits:
            if not q == -500:
                rm.append(q)
        
        wt.append(sum(rm)/len(rm))
        rt.append(cntt)
        cntt+=1

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    # plt.plot(rt , wt)
    # plt.show()

if __name__ == '__main__': 
    main()

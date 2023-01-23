import random
from binaryga import *
import statistics
import testfunctions
import datetime
import matplotlib.pyplot as plt


def bincode_ga(precis=None,
               lim_inf=None,
               lim_sup=None,
               num_vars=None,
               test_function=None,
               size_population=None,
               p_crosover=None,
               p_mutation=None,
               max_num_generations=None,
               elitism=None ,
               tol_time=None,
               plotting= False
               ):
    """
    Álgoritmo genético con codificación binaria
    """
 
    best_individuals = []*max_num_generations

    best_apt_per_generation = [0]*max_num_generations
    worst_apt_per_generation = [0]*max_num_generations
    mean_apt_per_generation = [0]*max_num_generations
    variance_apt_per_generation = [0]*max_num_generations

    BinIndiv.set_class_atr(test_function, lim_inf, lim_sup, num_vars, precis)
       
    if p_mutation == -1:
        p_mutation = 1/(RandBinGen.var_len*RandBinGen.dim)
    print(p_mutation)
        
    if plotting:
        plt.figure(1)
        globa
    num_generation = 0
    population = create_popul(size_population)

    while num_generation < max_num_generations:
        # selección
        pool = select_permutation_tournament(population)
        next_generation = pool.copy()

        # cruza
        for k in range(size_population):
            if random.random() <= p_crosover:
                i,j = random.randrange(0, size_population), random.randrange(0, size_population)
                next_generation[i], next_generation[j] = unif_cross(pool[i], pool[j])

        # mutación
        for individual in next_generation:
            mutate(individual)
        
        population = next_generation

        values = [individual.val for individual in population]
        worst_apt_per_generation[num_generation] = min(values)
        mean_apt_per_generation[num_generation] = statistics.fmean(values)
        best_apt_per_generation[num_generation] = min(values)

        num_generation = num_generation + 1
    return population, worst_apt_per_generation, mean_apt_per_generation, best_apt_per_generation


if __name__ == '__main__':


    parametros = { "precis": 4,
                   "lim_inf": testfunctions.rastring.vars_range[0],
                   'lim_sup': testfunctions.rastring.vars_range[1],
                   'num_vars': 2,
                   'test_function': testfunctions.rastring,
                   'size_population': 100,
                   'p_crosover': 0.5,
                   'p_mutation': 0.1,
                   'max_num_generations': 100,
                   'elitism': 0 ,
                   'tol_time': None
                   }
    result, worst_apt_per_generation, mean_apt_per_generation, best_apt_per_generation = bincode_ga(**parametros)
    result.sort( key = lambda x : x.val)
    for i in result[:10]:
        print(i)
    print( mean_apt_per_generation)


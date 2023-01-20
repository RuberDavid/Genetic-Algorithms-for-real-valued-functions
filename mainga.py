import random
from binaryga import *
import statistics
import testfunctions
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':


    parametros = { "precis": 4,
                   "lim_inf": testfunctions.rastring.vars_range[0],
                   'lim_sup': testfunctions.rastring.vars_range[1],
                   'num_vars': 2,
                   'test_function': testfunctions.rastring,
                   'size_population': 100,
                   'p_crosover': 0.5,
                   'p_mutation': 0.1,
                   'max_num_generations': 200,
                   'len_elite': 0 ,
                   'tol_time': None,
                   'plotting':False
                   }
    result, \
        worst_apt_per_generation, \
        mean_apt_per_generation, \
        best_apt_per_generation = bincode_ga(**parametros)

    result.sort(key=lambda x: x.val)

    for i in result[:10]:
        print(i)
    print( mean_apt_per_generation)


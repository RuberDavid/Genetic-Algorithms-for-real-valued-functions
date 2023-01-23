from binaryga import *
import statistics
import testfunctions
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    func_obj = testfunctions.rastring
    parametros = { "precis": 4,
                  "lim_inf": func_obj.vars_range[0],
                  'lim_sup': func_obj.vars_range[1],
                  'num_vars': 2,
                  'test_function': func_obj,
                  'size_population': 150,
                  'p_crosover': 0.5,
                  'p_mutation': -1, # por default : 1/ len(gen)
                  'max_num_generations': 200,
                  'len_elite': 0 ,
                  'tol_time': None,
                  'plotting': True
                 }

    last_generation, \
    worst_apt_per_generation, \
    mean_apt_per_generation, \
    best_apt_per_generation = bincode_ga(**parametros)



from binaryga import *
import statistics
import testfunctions
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    func_obj = testfunctions.rastring
    parametros = { "precis": 5,
                  "lim_inf": func_obj.vars_range[0],
                  'lim_sup': func_obj.vars_range[1],
                  'num_vars': 2,
                  'test_function': func_obj,
                  'size_population': 250,
                  'p_crosover': 0.5,
                  'p_mutation': None, # si e : 1/ len(gen)
                  'max_num_generations':100,
                  'len_elite': 1 ,
                  'tol_time': None,
                  'plotting': True
                 }

    last_generation, \
    worst_apt_per_generation, \
    mean_apt_per_generation, \
    best_apt_per_generation = bincode_ga(**parametros)

    for i in


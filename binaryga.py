import numpy as np
import random
import math
import testfunctions
import statistics
from typing import Callable
import matplotlib.pyplot as plt
import time

# Codificación

class RandBinGen(list): #TODO: se puede heredar de np.ndarray o de UserLIst
    lim_inf = None
    lim_sup = None
    dim = None
    precis = None
    var_len = None

    def set_class_atr(lim_inf: float,  # TODO : definir como classmethod
                      lim_sup: float,
                      dim :int,
                      precis : int):
        RandBinGen.lim_inf = lim_inf
        RandBinGen.lim_sup = lim_sup
        RandBinGen.dim     = dim
        RandBinGen.precis  = precis
        #TODO: sólo vale para dominios 'cuadrados', lind puede implementarse como una lista de rangos
        tam = math.ceil(math.log2((RandBinGen.lim_sup - RandBinGen.lim_inf) * 10 ** RandBinGen.precis))
        RandBinGen.var_len = tam

    def __init__(self, gen=None):  # TODO, que permita inicializar con un gen dado
        '''
        Inicializa un nuevo gen aleatorio
        '''
        super().__init__(self) # esto llama a init de la clase padre

        try:
            assert( self.lim_sup != None and
                    self.lim_inf != None and
                    self.precis != None and
                    self.dim != None)

        except Exception:
            raise Exception('Inicilizar primero la clase RandBinGen\n\
            Use RandBinGen.set_class_atr(lim_inf: float,lim_sup: float, dim :int, precis : int)\n\
            Por ejemplo: RandBinGen.set_class_atr(-5.2, 5.2, 2, 3)')

        if gen == None:
            self.extend([random.choice([1, 0])
                         for _ in range(self.var_len * self.dim)])  # TODO: convertir precisión
            return
        if len(gen) != RandBinGen.var_len * RandBinGen.dim:
            raise Exception('los tamaños no coinciden')
        self.extend(gen)
        return



    def expresa(self) -> list:

        '''expresa un gen(lista de ceros y unos) como par de coordenadas flotantes'''

        # divide en subcadenas de longitud l
        gen_split = [self[i * self.var_len : (i + 1) * self.var_len]
                     for i in range(self.dim)]

        # la lectua es de derecha a izquierda por cada subcadena de genes
        # convierte cada subcadena a entero
        var_x = [sum(cad[-(1+j)] * 2**j
                      for j in range(self.var_len))
                 for cad in gen_split]

        # tranforma al representación en punto fijo correspondiente (real)
        x = [self.lim_inf + x_i * (self.lim_sup - self.lim_inf) / (2 ** self.var_len - 1)
             for x_i in var_x]
        return x



class BinIndiv:
    # los siguientes atributos de clase no son fijos y deben de ser cambiados mediante el método set_class_atr
    func_obj = None

    def set_class_atr(func_obj: Callable,  # TODO: hacer un método de clase
                      lim_inf : float,
                      lim_sup: float,
                      dim: int,
                      precis: int):
        BinIndiv.func_obj = func_obj
        RandBinGen.set_class_atr(lim_inf, lim_sup, dim, precis)

    def __init__(self, nuevo_gen=None): #TODO dim debería ser un atributo del gen
        """ crea un nuevo indivuo con un genotipo, un fenotipo y un aptitud
            la aptitud es evaluada como
        """
        if  nuevo_gen == None:
            self._gen = RandBinGen()
        else:
            if not type(nuevo_gen) == RandBinGen:
                nuevo_gen = RandBinGen(nuevo_gen)
            self._gen = nuevo_gen
        self.fenotipo = self._gen.expresa() # TODO: esto podría ocupar mucha memoria o tiempo en alojarla
        self.val = BinIndiv.func_obj(*self.fenotipo)
        self.apt = None # TODO


    @property
    def gen(self):
        return self._gen

    @gen.setter
    def gen(self,new_gen):
        self._gen = RandBinGen(new_gen)
        self.fenotipo = self._gen.expresa()
        self.val = BinIndiv.func_obj(*self.fenotipo)

    def __str__(self):
        return f"gen = {self.gen}\nfenotipo = {self.fenotipo}\nvalor = {self.val}\n"

    #mutación en un punto
    def mutate(self):
        '''
        mutación de un bit
        '''
        n = len(self.gen)
        new_gen = self.gen.copy()
        
        rand_index = random.randrange(0, n)
        new_gen[rand_index] = (self.gen[rand_index] + 1) % 2 #
        self.gen = new_gen

def create_popul( size_pop :int ):
    return [BinIndiv() for _ in range(size_pop)]

############################################################################
# Método de mutación

def mutate(ind, p_mut):
        new_gen = ind.gen.copy()
        for i in range(len(new_gen)):
            if random.random() <= p_mut:
                new_gen[i] == (new_gen[i] + 1 )%2
        ind.gen = new_gen
#
# Métodos de selección

def select_permutation_tournament(pob: list, sent_opt='min') -> list:
    '''
    regresa una nueva generación mediante un torneo entre el pob y una permutación del pob
    '''
    if sent_opt not in {'max', 'min'}:
        raise Exception(sent_opt, " no es un argumneto válido")
    def compare(x, y):
        if sent_opt == 'max':
            return x > y
        else:
            return x < y
    permut = np.random.permutation(pob)

    return [pob[i] if compare(pob[i].val, permut[i].val) else permut[i] for i in range(len(pob))]


# TODO selección por ruleta
# TODO selección por SUS
##########################################################################
# Métodos de cruza

# TODO cruza uniforme
def unif_cross(parent1, parent2):
    n = len(parent1.gen)
    gen1 = [None]*n
    gen2 = [None]*n

    mask = random.choices([0,1],k=n)

    for i in range(n):
        if mask[i] == 1:
            gen1[i] = parent1.gen[i] # por la implementación como una propiedad, este proceso se vuelve ineficiente
            gen2[i] = parent2.gen[i]
        else:
            gen1[i] = parent2.gen[i]
            gen2[i] = parent1.gen[i]

    offspring1 = BinIndiv(gen1)
    offspring2 = BinIndiv(gen2)

    return offspring1, offspring2




############################################################################
# plotting helper function
def actulize_axes_helper(ax, data1, data2, param_dict):
    out = ax.

def bincode_ga(precis=None,
               lim_inf=None,
               lim_sup=None,
               num_vars=None,
               test_function=None,
               size_population=None,
               p_crosover=None,
               p_mutation=None,
               selection_func=None, # funciones de selección y cruza
               cross_func=None,
               mutate_func=None,
               max_num_generations=None,
               len_elite=None,
               tol_time=None,
               plotting=False
               ):
    """
    Álgoritmo genético con codificación binaria

    :returns population, worst_apt_per_generation, mean_apt_per_generation, best_apt_per_generation
    """
    elite = []*max_num_generations

    best_apt_per_generation = [0]*max_num_generations
    worst_apt_per_generation = [0]*max_num_generations
    mean_apt_per_generation = [0]*max_num_generations
    variance_apt_per_generation = [0]*max_num_generations

    BinIndiv.set_class_atr(test_function, lim_inf, lim_sup, num_vars, precis)
    
    if p_mutation == None:
        p_mutation = 1/(RandBinGen.var_len*RandBinGen.dim)
    print("p_mutation", p_mutation)

    num_generation = 0
    population = create_popul(size_population)

    if plotting:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle(f"Minimización de la función { test_function.__name__} con AG\n"
                  f"p_mut = {p_mutation}\n"
                  f"p_cruz = {p_crosover}\n")

        plt.xlim(test_function.vars_range)
        plt.ylim(test_function.vars_range)

        global_minima = test_function.minima

        plt.scatter(*zip(*global_minima), marker='X', color='red',zorder=max_num_generations+1)
        plt.scatter(*zip(*[ind.fenotipo for ind in population]),
                    marker='.', c=size_population*[num_generation],
                    vmin = 0,vmax=max_num_generations, zorder=num_generation)

    

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
            mutate(individual, p_mutation) #

        population = next_generation
        if elite:
            elite[num_generation] = max(population, key = lambda x:x.val)

        values = [individual.val for individual in population]
        worst_apt_per_generation[num_generation] = min(values)
        mean_apt_per_generation[num_generation] = statistics.fmean(values)
        best_apt_per_generation[num_generation] = min(values)
        
        if plotting:
            phenotype_last_gen = [ind.fenotipo for ind in population]

            plt.title(f"Minimización de la función { test_function.__name__} con AG\n"
                      f"p_mut = {p_mutation}\n"
                      f"p_cruz = {p_crosover}\n"
                      f"generación = {num_generation}")
            plt.scatter(*zip(*phenotype_last_gen),
                        marker='.', c=size_population*[num_generation],
                        vmin = 0,vmax=max_num_generations, zorder=num_generation)

            plt.pause(0.025)
            #plt.clf()
            
        num_generation = num_generation + 1

    if plotting:
        plt.show()
    return population, worst_apt_per_generation, mean_apt_per_generation, best_apt_per_generation



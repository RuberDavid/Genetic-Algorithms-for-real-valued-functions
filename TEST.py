from binaryga import *

# def AG_testfunc_bincode( func_obj,
#                          len_gen,
#                          pob_size,
#                          p_cruz,
#                          p_mut,
#                          tol=100):
#
#     pob = [BinIndiv(rand_bin_gen(len_gen), func_obj)
#            for _ in range(pob_size)]
#
#     for _ in range(tol):
#         # pool
#         # cruza
#         # muta



if __name__ == '__main__':
    LEN_GEN = 10
    POP_SIZE = 10
    CROSS_P = 0.5
    MUT_PROB = 0.2
    NUM_GENER = 50
    Elitism = 0 # porcentaje de elitismo
    TIME_TOL = 1 # minutos

    liminf = -5.12
    limsup = 5.12

    ind1 = BinIndiv( rand_bin_gen(LEN_GEN), liminf, limsup, rastring )
    ind2 = BinIndiv( rand_bin_gen(LEN_GEN), liminf, limsup, rastring )

    a,b = unif_cross(ind1, ind2)

    print(a)

    a.mutate()
    print(a)



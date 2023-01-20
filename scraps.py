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

DIM = 3
indiv1 = BinIndiv(rand_bin_gen(10, dim=DIM), -5.12, 5.12, rastring, dim=DIM)
indiv2 = BinIndiv(rand_bin_gen(10,dim=DIM), 5.12, -5.12, rastring, dim=DIM)

popul = create_popul(10, -5.12, 5.12, rastring, 2, 100)

parameters = {"gen_precis":     10,
              "lim_inf":        -5.12,
              "lim_sup":        5.12,
              "test_function":  testfunctions.rastring,
              "num_vars":       2,
              "size_population":7}

try:
    assert( 1 > 0 )
    l = [] +1
except Exception:
    raise Exception('ocurrió error ')

RandBinGen.set_class_atr(-5,5,2,4)
gen = RandBinGen()
gen2 = RandBinGen([random.choice([0,1]) for _ in range(RandBinGen.var_len*RandBinGen.dim)])

BinIndiv.set_class_atr(testfunctions.rastring,-5.12, 5.12, 2, 4)
ind = BinIndiv()
print(ind.gen)
ind2 = BinIndiv()
ind3 = BinIndiv()

ind3.gen = RandBinGen( [random.choice([0,1]) for _ in range(ind.gen.dim * ind.gen.var_len)])
ind4 = BinIndiv([random.choice([0,1]) for _ in range(ind.gen.dim * ind.gen.var_len)])

off1, off2 =


n = 100
x = np.linspace(-5.12, 5.12, n)
y = np.linspace(-5.12, 5.12, n)
X, Y = np.meshgrid(x, y)

plt.axes([0.025, 0.025, 0.95, 0.95])

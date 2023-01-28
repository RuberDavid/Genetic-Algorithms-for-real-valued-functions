import math
import numpy as np


##
def rastring(*args)->float:
    '''
    función Rastring para n variables
    '''
    if not all( [ abs( x_i) <= 5.12 for x_i in args ] ):# TODO : cambiar esta intefaz
        raise ValueError('all x_i should be in [-5.12, 5.12]')
    A = 10
    n = len(args)
    return A*n + math.fsum([x_i**2 - A*math.cos(2*math.pi*x_i) for x_i in args ])



rastring.vars_range = (-5.12,5.12)
rastring.minima = [ (0,0) ] # TODO: definir para mayor dimensión


def rosenbrock(*args):
    x = args
    #if not all([ abs(x_i) <= 5 for x_i in x ]):
    #    raise ValueError
    return math.fsum([100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1) ] )


rosenbrock.vars_range = (-100,100)
rosenbrock.minima = [ (1,1) ]


def himmelblau(*args):
    x,y = args[0], args[1]
    if not (abs(x) <= 5 and abs(y) <= 5 ):
        raise ValueError
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

himmelblau.vars_range = (-5,5)
himmelblau.minima = [(3.0,  2.0),
                     (-2.805118, 3.131312),
                     (-3.779310, -3.283186),
                     (3.584428, -1.848126)]

def eggholder(*args):
    x,y = args[0], args[1]
    if not (abs(x) <= 512 and abs(y) <= 512 ):
        raise ValueError
    return -(y+47)*math.sin(math.sqrt(abs(x/2 + (y+47))) - x*math.sin(math.sqrt(abs(x-(y+47)))))

eggholder.vars_range = (-512,512)
eggholder.minima = [(512, 404.2319)]
# TODO: mínimos

# TODO: to give arrbitrary function
############################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 100
    xy = np.linspace(1, 100, 5)

    y = rastring(x)
    X, Y = np.meshgrid(x, y)

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.contourf(X, Y, rastring(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, rastring(X, Y), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)

    plt.xticks([])
    plt.yticks([])
    plt.show()

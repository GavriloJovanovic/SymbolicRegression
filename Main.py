import numpy as np
import SymbolicRegression
import pdb
import random


def f(x):
    return x * x * x * x - 4 * x * x * x + x * x - 5 * x + 1

try:
    
    niz = [round((x - 30) / 10, 3) for x in range(0, 61, 2)]
    trening = [[niz[i],f(niz[i])] for i in range(0,31)]
    drvo = SymbolicRegression.GP(trening, 1000, 40, 10, 300, 0.1)
    drvo.printPopulation()
    print("POCINJEMO")
    print("============================================================")
    # pdb.set_trace()
    drvo.GP()
    print("POBEDNIK JE")
    print(drvo.population[0][0].stringNode() + " a njegov fitness je: " + str(drvo.population[0][1]))
    '''
    niz = [round((x - 30) / 10, 3) for x in range(0, 61, 2)]
    trening = [[niz[i],f(niz[i])] for i in range(0,31)]
    drvo = Node.GP(trening, 2, 1000, 10, 50, 0.2)
    drvo.onlyMutate()
    '''

except IOError:
    print("Greska")